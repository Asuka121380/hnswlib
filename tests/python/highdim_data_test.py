"""Parquet ingestion, provenance, duplicate groups, review gates and exact-GT tests."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.datasets import prepare_highdim as hd
from scripts.datasets import highdim_controls as hc
from scripts.datasets import final_study_data as data
from scripts.edge_estimation.final_study.common import identity, seal, unseal, write, load
from scripts.edge_estimation.final_study.query_contract import read_vecs, read_ids


def vector(i, d):
    return np.random.default_rng(i + 7712).normal(size=d).astype(np.float64).tolist()


def fixture(root, name="bioasq1024", base_count=32, query_count=16, records=None):
    """Create a clearly synthetic lock at the REAL profile dimension; no real queries."""
    spec = hd.profile(name)
    root.mkdir(parents=True, exist_ok=True)
    d = spec["dimension"]
    if records is None:
        if name == "bioasq1024":
            records = {"base": [{"id": i, "emb": vector(i, d)} for i in range(base_count)],
                       "query": [{"id": i, "emb": vector(i + base_count, d)} for i in range(query_count)]}
        else:
            records = {"corpus": [{"_id": f"entity-{i}", "title": f"title-{i}", "text": f"text-{i}",
                        spec["vector_column"]: vector(i, d)} for i in range(base_count + query_count)]}
    entries = []
    for index, (role, values) in enumerate(records.items()):
        actual_role = role.split("-")[0]
        fields = [pa.field(spec["id_column"], pa.int64() if name == "bioasq1024" else pa.string()),
                  pa.field(spec["vector_column"], pa.list_(pa.float64()))]
        if name != "bioasq1024":
            fields += [pa.field("title", pa.string()), pa.field("text", pa.string())]
        path = root / f"{role}-{index:03d}.parquet"
        pq.write_table(pa.Table.from_pylist(values, schema=pa.schema(fields)), path, row_group_size=3)
        entries.append({**identity(path), "role": actual_role, "relative_path": path.name, "url": "synthetic://fixture"})
    path = root / "source-lock.json"
    seal(path, {"schema_version": 1, **{k: spec[k] for k in ("profile", "dataset_id", "dimension", "id_column", "vector_column")},
                "revision": spec.get("revision"), "synthetic_only": True, "raw_files": entries})
    return path


def convert(lock, out, name="bioasq1024", batch=3, holdout=None):
    with redirect_stdout(io.StringIO()):
        hd.convert(NS(source_lock=str(lock), profile=name, out=str(out), batch_size=batch,
                      holdout_count=holdout, holdout_seed=20261010, allow_synthetic=True, normalize=hd.NORMALIZATION))
    return unseal(out / "conversion_manifest.json")


def reviewed_controls(root, converted, counts=(6, 4, 6, 2), reviewed=True):
    manifest = converted / "conversion_manifest.json"
    doc = unseal(manifest)
    review = root / "review.json"
    write(review, {"dataset_id": doc["dataset_id"], "source_sha256": doc["outputs"]["query_candidates.fvecs"]["sha256"],
                   "range": [0, doc["n_query_candidates"]], "status": "unused_confirmed", "review_complete": True,
                   "reviewer": "synthetic fixture test", "basis": "Fresh synthetic fixture generated in this isolated test."})
    out = root / ("controls-reviewed" if reviewed else "controls-pending")
    with redirect_stdout(io.StringIO()):
        hc.data_controls(NS(conversions=[str(manifest)], out=str(out), allow_synthetic=True,
                            synthetic_counts=counts, review_evidence=[f"{doc['dataset_id']}={review}"] if reviewed else [], history_root=[]))
    return out


class HighdimDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="highdim-unit-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_probe_reads_only_bounded_footer_and_rejects_full_response(self):
        lock = fixture(self.root/'raw')
        raw = Path(unseal(lock)['raw_files'][0]['path']).read_bytes()
        requests = []
        def serve(request, timeout):
            count = int(request.get_header('Range').split('-')[-1])
            requests.append(count)
            response = io.BytesIO(raw[-count:])
            response.status = 206
            response.headers = {'Content-Range': f'bytes {len(raw)-count}-{len(raw)-1}/{len(raw)}'}
            return response
        with patch.object(hd.urllib.request, 'urlopen', side_effect=serve):
            metadata = hd.remote_metadata('https://fixture.invalid/data.parquet')
        self.assertEqual(metadata['rows'], 32)
        self.assertEqual(requests, [8, struct.unpack('<I', raw[-8:-4])[0] + 8])
        self.assertLess(sum(requests), len(raw))
        response = io.BytesIO(raw)
        response.status = 200
        response.headers = {}
        with patch.object(hd.urllib.request, 'urlopen', return_value=response):
            with self.assertRaisesRegex(ValueError, 'refusing full download'):
                hd.small_request('https://fixture.invalid/data.parquet', 'bytes=-8', 8)
        response = io.BytesIO(b'123456789')
        response.status = 206
        response.headers = {}
        with patch.object(hd.urllib.request, 'urlopen', return_value=response):
            with self.assertRaisesRegex(ValueError, 'exceeds probe budget'):
                hd.small_request('https://fixture.invalid/data.parquet', 'bytes=-8', 8)

    def test_local_lock_and_download_publish_complete_identity(self):
        source = fixture(self.root/'fixture')
        entries = unseal(source)['raw_files']
        # Populate every canonical BioASQ source name without external requests.
        raw_dir = self.root/'canonical'
        raw_dir.mkdir()
        spec = hd.profile('bioasq1024')
        payloads = {}
        for entry in spec['files']:
            data = Path(entries[1 if entry['role'] == 'query' else 0]['path']).read_bytes()
            (raw_dir/entry['relative_path']).write_bytes(data)
            payloads[hd.source_url(spec, entry['relative_path'])] = data
        local = self.root/'local-lock.json'
        hd.lock_local(NS(profile='bioasq1024', raw_dir=raw_dir, out=local))
        hd.load_lock(local, 'bioasq1024')
        with self.assertRaisesRegex(ValueError, 'exists'):
            hd.lock_local(NS(profile='bioasq1024', raw_dir=raw_dir, out=local))
        def serve(url, timeout):
            response = io.BytesIO(payloads[url])
            response.headers = {'Content-Length': str(len(payloads[url]))}
            return response
        destination = self.root/'download'
        with patch.object(hd.urllib.request, 'urlopen', side_effect=serve), redirect_stdout(io.StringIO()):
            hd.download(NS(profile='bioasq1024', out=destination))
        downloaded, _ = hd.load_lock(destination/'source-lock.json', 'bioasq1024')
        self.assertEqual(len(downloaded['raw_files']), 3)
        for entry in downloaded['raw_files']:
            self.assertEqual(Path(entry['path']).parent, destination)
            self.assertEqual(entry['sha256'], identity(raw_dir/entry['relative_path'])['sha256'])
        with patch.object(hd.urllib.request, 'urlopen') as request:
            with self.assertRaises(FileExistsError):
                hd.download(NS(profile='bioasq1024', out=destination))
            request.assert_not_called()

    def test_truncated_download_never_publishes_success(self):
        response = io.BytesIO(b'truncated')
        response.headers = {'Content-Length': '500'}
        destination = self.root/'download'
        with patch.object(hd.urllib.request, 'urlopen', return_value=response), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'length mismatch'):
                hd.download(NS(profile='bioasq1024', out=destination))
        self.assertFalse(destination.exists())
        partials = list(self.root.glob('.download.partial-*'))
        self.assertEqual(len(partials), 1)
        self.assertFalse((partials[0]/'source-lock.json').exists())

    def test_bio_overlap_dedup_invalid_and_row_labels(self):
        d = 1024
        a, b, c = vector(0,d), vector(1,d), vector(2,d)
        records = {"base": [{"id": 10, "emb": a}, {"id": 20, "emb": b}, {"id": 30, "emb": [0.0]*d}],
                   "query": [{"id": 10, "emb": c}, {"id": 11, "emb": c}, {"id": 12, "emb": a},
                             {"id": 13, "emb": None}, {"id": 14, "emb": [float('nan')]*d},
                             {"id": 15, "emb": [None]*d}, {"id": 16, "emb": [0.0]*d}]}
        doc = convert(fixture(self.root/'raw', records=records), self.root/'out')
        self.assertEqual((doc['n_base'],doc['n_query_candidates']), (2,1))
        self.assertEqual(doc['exclusions'], {'zero_norm':2,'duplicate_query':1,'base_overlap':1,'null_vector':1,'nonfinite_vector':1,'null_element':1})
        q = read_vecs(self.root/'out/query_candidates.fvecs',dimension=d)
        self.assertTrue(np.allclose(np.linalg.norm(q,axis=1),1,atol=1e-6))
        origin = json.loads((self.root/'out/query_row_to_source_id.jsonl').read_text())
        self.assertEqual(origin['source_id'],10)  # Same numeric ID as base is not leakage.
        self.assertEqual(origin['row'],0)
        self.assertFalse(doc['history_reviewed'])

    def test_batch_size_invariance_and_no_overwrite(self):
        lock = fixture(self.root/'raw')
        a = convert(lock,self.root/'a',batch=1)
        b = convert(lock,self.root/'b',batch=7)
        for name in a['outputs']:
            self.assertEqual(a['outputs'][name]['sha256'],b['outputs'][name]['sha256'])
        before = identity(self.root/'a/conversion_manifest.json')
        with self.assertRaises(FileExistsError):
            convert(lock,self.root/'a')
        self.assertEqual(before,identity(before['path']))

    def test_wrong_dimension_fails_atomically(self):
        records={'base':[{'id':0,'emb':[1.0,2.0]}],'query':[{'id':1,'emb':vector(1,1024)}]}
        with self.assertRaisesRegex(ValueError,'dimension'):
            convert(fixture(self.root/'raw',records=records),self.root/'out')
        self.assertFalse((self.root/'out').exists())
        self.assertTrue(list(self.root.glob('.out.partial-*')))

    def test_conflicting_id_rejected(self):
        records={'base':[{'id':1,'emb':vector(1,1024)},{'id':1,'emb':vector(2,1024)}],
                 'query':[{'id':2,'emb':vector(3,1024)}]}
        with self.assertRaisesRegex(ValueError,'conflicting'):
            convert(fixture(self.root/'raw',records=records),self.root/'out')

    def test_changed_raw_file_rejected(self):
        lock=fixture(self.root/'raw')
        entry=unseal(lock)['raw_files'][0]
        with Path(entry['path']).open('ab') as f: f.write(b'corrupt')
        with self.assertRaisesRegex(ValueError,'identity mismatch'):
            convert(lock,self.root/'out')

    def test_pinned_revision_and_complete_sources(self):
        lock=fixture(self.root/'raw',name='dbpedia1536')
        doc=unseal(lock);doc['revision']='main';seal(lock,doc)
        with self.assertRaisesRegex(ValueError,'pinned revision'):
            convert(lock,self.root/'out',name='dbpedia1536',holdout=8)
        doc['revision']=hd.profile('dbpedia1536')['revision'];doc['synthetic_only']=False;seal(lock,doc)
        with self.assertRaisesRegex(ValueError,'published shards'):
            convert(lock,self.root/'out',name='dbpedia1536',holdout=8)

    def test_synthetic_requires_explicit_flag(self):
        lock=fixture(self.root/'raw')
        with self.assertRaisesRegex(ValueError,'synthetic'):
            hd.load_lock(lock,'bioasq1024')
        convert(lock,self.root/'out')
        with self.assertRaisesRegex(ValueError,'synthetic'):
            hd.verify_conversion(self.root/'out/conversion_manifest.json')

    def test_dbpedia_transitive_groups_holdout_multishard(self):
        name='dbpedia1536';d=1536;col=hd.profile(name)['vector_column']
        def rec(i, v, title, text):
            return {'_id':f'id-{i}',col:v,'title':title,'text':text}
        first=[rec(0,vector(0,d),'A','T'),rec(1,vector(1,d),'different title','T')]
        second=[rec(2,vector(1,d),'B','U')]+[rec(i,vector(i,d),str(i),str(i)) for i in range(3,24)]
        lock=fixture(self.root/'raw',name=name,records={'corpus-a':first,'corpus-b':second})
        a=convert(lock,self.root/'a',name=name,holdout=8,batch=1)
        b=convert(lock,self.root/'b',name=name,holdout=8,batch=9)
        self.assertEqual((a['n_base'],a['n_query_candidates']), (14,8))
        self.assertEqual(a['exclusions']['duplicate_group'],2)
        for file in a['outputs']:
            self.assertEqual(a['outputs'][file]['sha256'],b['outputs'][file]['sha256'])
        origins={}
        for role in ('base','query'):
            origins[role]=[json.loads(line) for line in (self.root/f'a/{role}_row_to_source_id.jsonl').read_text().splitlines()]
        self.assertFalse({r['source_id'] for r in origins['base']} & {r['source_id'] for r in origins['query']})
        self.assertFalse({r['vector_sha256'] for r in origins['base']} & {r['vector_sha256'] for r in origins['query']})
        self.assertFalse({r['text_sha256'] for r in origins['base']} & {r['text_sha256'] for r in origins['query']})
        self.assertNotIn('id-1',{r['source_id'] for rows in origins.values() for r in rows})
        self.assertNotIn('id-2',{r['source_id'] for rows in origins.values() for r in rows})

    def test_dbpedia_conflicting_title_under_same_id_rejected(self):
        name = 'dbpedia1536'
        col = hd.profile(name)['vector_column']
        records = {'corpus': [{'_id':'same', 'title':title, 'text':'same body', col:vector(0,1536)}
                               for title in ('first', 'conflicting')]}
        lock = fixture(self.root/'raw', name=name, records=records)
        with self.assertRaisesRegex(ValueError, 'conflicting duplicate source ID'):
            convert(lock, self.root/'out', name=name, holdout=1)

    def test_optional_3072_profile(self):
        lock=fixture(self.root/'raw',name='dbpedia3072',base_count=4,query_count=4)
        doc=convert(lock,self.root/'out',name='dbpedia3072',holdout=3)
        self.assertEqual(doc['dimension'],3072)
        self.assertEqual(read_vecs(self.root/'out/query_candidates.fvecs').shape,(3,3072))

    def test_arrow_null_id_and_missing_columns(self):
        lock=fixture(self.root/'raw',records={'base':[{'id':None,'emb':vector(1,1024)}],
                                           'query':[{'id':2,'emb':vector(2,1024)}]})
        with self.assertRaisesRegex(ValueError,'source ID'):
            convert(lock,self.root/'out')
        lock2=fixture(self.root/'raw2')
        doc=unseal(lock2);e=doc['raw_files'][0]
        pq.write_table(pa.table({'id':[1]}),e['path']);e.update(identity(e['path']));seal(lock2,doc)
        with self.assertRaisesRegex(ValueError,'missing source columns'):
            convert(lock2,self.root/'out2')

    def test_normalization_zero_sign_and_extreme_finite(self):
        for values in ([1.0,-0.0]+[0.0]*1022, [1e300]*1024, [1e-300]*1024):
            scalar=pa.array([values],type=pa.list_(pa.float64()))[0]
            v,reason=hd.normalized(scalar,1024)
            self.assertIsNone(reason)
            self.assertAlmostEqual(float(np.linalg.norm(v)),1.0,places=6)
            self.assertFalse(np.signbit(v[v==0]).any())

    def test_capacity_rules(self):
        for n,expected in [(3102,(500,500,2000,100)),(2750,(500,500,1750,100))]:
            self.assertEqual(hc.split_counts({'profile':'bioasq1024','n_query_candidates':n}),expected)
        with self.assertRaisesRegex(ValueError,'2500'):
            hc.split_counts({'profile':'bioasq1024','n_query_candidates':2499})
        with self.assertRaisesRegex(ValueError,'10000'):
            hc.split_counts({'profile':'dbpedia1536','n_query_candidates':9999})

    def test_pending_history_cannot_materialize(self):
        convert(fixture(self.root/'raw'),self.root/'converted')
        controls=reviewed_controls(self.root,self.root/'converted',reviewed=False)
        self.assertFalse(unseal(controls/'controls_manifest.json')['ready_for_history_audit'])
        data.audit_queries(NS(registry=controls/'registry.highdim.json',history_roots=controls/'history.highdim.json',out=self.root/'audit'))
        with self.assertRaisesRegex(ValueError,'insufficient eligible'):
            data.materialize(NS(audit=self.root/'audit/audit.json',dataset='bioasq1024_cos_v1',
                                split_spec=controls/'splits.highdim.json',seed=20261008,out=self.root/'data'))

    def test_bad_review_and_manifest_tampering(self):
        convert(fixture(self.root/'raw'),self.root/'converted')
        bad=self.root/'review.json';write(bad,{'review_complete':True})
        with self.assertRaisesRegex(ValueError,'explicitly bind'):
            hc.review_for(bad,'bioasq1024_cos_v1','abc',16)
        p=self.root/'converted/base.fvecs'
        with p.open('ab') as f:f.write(b'bad')
        with self.assertRaisesRegex(ValueError,'identity mismatch'):
            hd.verify_conversion(self.root/'converted/conversion_manifest.json',True)

    def test_conversion_to_independent_gt_pipeline(self):
        convert(fixture(self.root/'raw',base_count=128),self.root/'converted')
        controls=reviewed_controls(self.root,self.root/'converted')
        data.inventory(NS(registry=controls/'registry.highdim.json',out=self.root/'inventory'))
        data.audit_queries(NS(registry=controls/'registry.highdim.json',history_roots=controls/'history.highdim.json',out=self.root/'audit'))
        data.materialize(NS(audit=self.root/'audit/audit.json',dataset='bioasq1024_cos_v1',
                            split_spec=controls/'splits.highdim.json',seed=20261008,out=self.root/'data'))
        manifest=self.root/'data/dataset_manifest.json'
        for split in ('dev','select','test'):
            data.exact_gt(NS(dataset_manifest=manifest,split=split,k=100,base_block=64,query_block=4,threads=1,backend='float64'))
        with redirect_stdout(io.StringIO()):
            data.validate(NS(dataset_manifest=manifest,float64_check_queries=4,seed=20261008,base_block=64))
        doc=unseal(self.root/'data/dataset_contract.json')
        self.assertTrue(doc['synthetic_only'])
        self.assertEqual(doc['preprocessing_manifest'],identity(self.root/'converted/conversion_manifest.json'))
        ids=[set(read_ids(s['ids']['path'])) for s in doc['splits'].values()]
        self.assertEqual(len(set.union(*ids)),16)
        self.assertEqual(sum(map(len,ids)),16)
        base=read_vecs(doc['base']['path']);query=read_vecs(doc['query_pool']['path'])
        labels=read_vecs(doc['splits']['test']['gt']['path'],'i')
        qids=read_ids(doc['splits']['test']['ids']['path'])
        expected=np.argsort(np.sum((base.astype(float)-query[qids[-1]].astype(float))**2,axis=1))[:10]
        self.assertEqual(labels[-1,:10].tolist(),expected.tolist())

    def test_protocol_preserves_budget_and_excludes_supplement(self):
        base=ROOT/'configs/edge_estimation/final_study/protocol.json'
        grids=self.root/'grids.json';write(grids,{d:[100,200,400] for d in hc.NEW_DATASETS})
        out=self.root/'v2.json'
        with redirect_stdout(io.StringIO()):
            hc.protocol_controls(NS(base_protocol=base,grids=grids,out=out,pilot=False))
        p=load(out)
        self.assertEqual(p['training'],load(base)['training'])
        self.assertEqual(p['selection']['required_datasets'],list(hc.MAIN_DATASETS))
        self.assertEqual(set(p['development']['ef_grids']),set(hc.MAIN_DATASETS))
        with self.assertRaisesRegex(ValueError,'explicit reviewed'):
            hc.protocol_controls(NS(base_protocol=base,grids=None,out=self.root/'no.json',pilot=False))
        with self.assertRaisesRegex(ValueError,'exists'):
            hc.protocol_controls(NS(base_protocol=base,grids=grids,out=out,pilot=False))

    def test_environment_wrapper_order_and_quoting(self):
        out=self.root/'env.sh'
        hc.environment_controls(NS(v1='/project/old space',study='/project/new space',repo='/project/repo',out=out))
        text=out.read_text()
        self.assertLess(text.index('source '),text.index('export STUDY='))
        self.assertIn("source '/project/old space/control/env.sh'",text)
        self.assertIn('INPUT_ROOT="$V1/$ds"',text)
        with self.assertRaisesRegex(ValueError,'replace'):
            hc.environment_controls(NS(v1='/same',study='/same',repo='/repo',out=self.root/'bad.sh'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
