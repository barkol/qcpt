import sys, time, os, json, resource, statistics as st
"""Wall and CPU time of stage 1 (ErrorBudgetSelector.select, certificate included) for the nine circuits of the comparison with the
Claude loop (median of 7 runs after one warm-up; single-threaded BLAS: run with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1).
Usage: stage1_time.py [OUT.json]   ->  paper/figures_resynthesis/data_in/new/stage1_times.json"""
root=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".."))
sys.path[:0]=[root,root+"/experiments",root+"/experiments/race",root+"/experiments/revlib"]
os.environ["R"]=root
import portfolio as PF
from toffoli_optimizer.core.decomposition_selector import ErrorBudgetSelector
nine=['controlled_adder_2b','controlled_adder_3b','half_uncomputed_4c','grover_oracle_mcx4','mod_incr_3b','ripple_carry_adder_2b','rl_4mod5-bdd_287','rl_alu-bdd_288','rl_rd53_138']
out={}
for nm in nine:
    qc,pins,n,inputs=PF.load(nm)
    ErrorBudgetSelector().select(qc,pinned_zero=pins)             # warm-up
    W=[];C=[]
    for _ in range(7):
        c0=resource.getrusage(resource.RUSAGE_CHILDREN); t0=time.perf_counter(); p0=time.process_time()
        ErrorBudgetSelector().select(qc,pinned_zero=pins)
        W.append(time.perf_counter()-t0); c1=resource.getrusage(resource.RUSAGE_CHILDREN)
        C.append(time.process_time()-p0+(c1.ru_utime+c1.ru_stime)-(c0.ru_utime+c0.ru_stime))
    out[nm]=dict(wall=st.median(W),cpu=st.median(C),n=n,gates=len(qc.data))
    print("%-22s n=%d gates=%3d  wall %.3f s  cpu %.3f s" % (nm,n,len(qc.data),out[nm]["wall"],out[nm]["cpu"]),flush=True)
json.dump(out,open(sys.argv[1] if len(sys.argv)>1 else os.path.join(root,"paper","figures_resynthesis","data_in","new","stage1_times.json"),"w"),indent=1)
