"""Independent D=960/M32b8 layout oracle, including zero-edge fallback."""
import csv
import io
import json
import struct
from pathlib import Path
import numpy as np
from scripts.edge_estimation.contracts import EventRecord, write_events, sha256_file
from scripts.edge_estimation.ivf_train import dump, inventory


def dimension_oracle(root, native, run):
    root.mkdir()
    rng=np.random.default_rng(71)
    d,m,k,nc,b=960,32,256,4,128
    q=rng.normal(0,.1,(3,d)).astype("<f4"); q[0]=0
    c=rng.normal(0,.1,d); edge=rng.normal(0,.05,d); length=np.linalg.norm(edge)
    books=rng.normal(0,.01,(m,k,d//m)).astype("<f4")
    centers=rng.normal(0,.01,(nc,d)).astype("<f4")
    codes=rng.integers(0,k,m,dtype=np.uint8)
    rotation=np.roll(np.eye(d,dtype="<f4"),1,axis=1)
    projection=rng.normal(size=(b,d)).astype("<f4")
    with (root/"queries.fvecs").open("wb") as out:
        for query in q:
            out.write(struct.pack("<I",d)); query.tofile(out)
    identity=bytes(range(32)); events=[]
    for query_id in (0,2,1):
        dcur=float(np.sum((q[query_id].astype(float)-c)**2))
        def event(kind,flags,layer,**kw):
            return EventRecord(kind,flags,layer,len(events),query_id,**kw)
        events.append(event(1,0,-1))
        events.append(event(2,0,-1,expansion_id=0,source_id=0,source_degree=2,d_current=dcur))
        for edge_id in (0,1):
            events.append(event(3,7,0,expansion_id=0,source_id=0,target_id=edge_id+1,source_degree=2,
                                neighbor_slot=edge_id,edge_id=edge_id,d_current=dcur,threshold_before=10))
        events.append(event(5,0,-1))
    write_events(root/"events.bin",d,identity,events)
    errors={}
    for method in ("pq","opq","pq_qjl"):
        dest=root/method; dest.mkdir()
        books.tofile(dest/"codebook.f32le"); centers.tofile(dest/"centers.f32le")
        h=np.concatenate([books[j,codes[j]].astype(float) for j in range(m)])
        if method=="opq":
            rotation.tofile(dest/"rotation.f32le"); h=h@rotation.astype(float)
        h+=centers[2].astype(float)
        offset=length**2+2*length*np.dot(c,h)
        residual=edge-length*h
        scale=np.linalg.norm(residual)*np.sqrt(np.pi/2)/b
        signs=np.where(projection.astype(float)@residual>=0,1.,-1.)
        if method=="pq_qjl":
            offset+=2*np.dot(c,residual); projection.tofile(dest/"projection.f32le")
        data=struct.pack("<ddI",length,offset,2)+codes.tobytes()
        if method=="pq_qjl":
            data+=struct.pack("<d",scale)+np.packbits(signs>0,bitorder="little").tobytes()
        stride=len(data)
        (dest/"edges.bin").write_bytes(data+struct.pack("<d",0)+data[8:])
        dump(dest/"manifest.json",{"files":inventory(dest),"fixture":True})
        cfg={"format":"uq-ivf-edge/1","backend":"ivf_"+method,"dimension":d,"m":m,"nbits":8,
             "nlist":nc,"qjl_bits":b if method=="pq_qjl" else 0,"record_size":stride,"edge_count":2,
             "rotation":int(method=="opq"),"coverage":"full_graph","catalog_identity":identity.hex(),
             "manifest_sha256":sha256_file(dest/"manifest.json")}
        for name,entry in inventory(dest).items(): cfg[name+"_sha256"]=entry["sha256"]
        (dest/"native.cfg").write_text("".join(f"{key}={value}\n" for key,value in cfg.items()))
        (dest/"complete.sha256").write_text(sha256_file(dest/"native.cfg")+"\n")
        records=list(csv.DictReader(io.StringIO(run([native,"scores-artifact",dest,root/"events.bin",root/"queries.fvecs"]))))
        maximum=0
        for row in records:
            event=events[int(row["event_id"])]; value=float(row["estimate"])
            if event.edge_id==1:
                assert np.isnan(value); continue
            query=q[event.query_id].astype(float)
            expected=event.d_current+length**2-2*length*np.dot(query-c,h)
            if method=="pq_qjl": expected+=-2*scale*np.dot(projection.astype(float)@query,signs)+2*np.dot(c,residual)
            np.testing.assert_allclose(value,expected,atol=1e-4,rtol=1e-5)
            maximum=max(maximum,abs(value-expected))
        parity=json.loads(run([native,"validate-batch-artifact",dest,root/"events.bin",root/"queries.fvecs",2]))
        assert parity["valid"] and parity["nonfinite_mismatch_count"]==0
        errors[method]=maximum
    return {"max_errors":errors,"dimension":d,"m":m,"nbits":8,"qjl_bits":b,"zero_edge_fallback":True}
