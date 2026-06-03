# k8s-netperf-viz

A Streamlit dashboard (`k8snetperf_dashboard.py`) that visualizes output from [k8s-netperf](https://github.com/cloud-bulldozer/k8s-netperf), a Kubernetes network performance benchmarking tool. The entire app is a single Python file.

## Running the App

```bash
pip install streamlit pandas plotly numpy
streamlit run k8snetperf_dashboard.py (--server.address 0.0.0.0 --server.port 8080)
```

The app expects two types of input: The URL of the job, with `build-log.txt`, or `k8snetperf_raw.csv` placed in the working directory. Users can upload a CSV via the sidebar.

The job's URL is roughly:

```
https://xxx.yyy.zzz.com/(...)/artifacts/daily-virt-6nodes/oxxxxxxt-qe-nyyyyyyk-perf/build-log.txt
```

The CSV format from k8s-netperf should be as follows (example data snippet):

```
Role,Driver,Profile,Same node,Host Network,VM mode,Service,External Server,UDN Info,Bridge Info,Duration,Parallelism,# of Samples,Message Size,Burst,Confidence metric - low,Confidence metric - high,Idle CPU,User CPU,System CPU,IOWait CPU,Steal CPU,SoftIRQ CPU,IRQ CPU
Server,netperf,TCP_STREAM,false,false,false,false,false,,,30,1,5,64,0,1069.8736719280953,1121.4103280719048,99.253001,0.373438,0.159549,0.061409,0.000000,0.124752,0.013219
Client,netperf,TCP_STREAM,false,true,false,false,false,,,30,2,5,64,0,2117.5115266577513,2361.624473342249,98.495114,0.532688,0.776910,0.000298,0.000000,0.170427,0.013839
(...)
```

The parser (`split_raw_csv`) identifies table boundaries by detecting header values (`Role`, `Type`, `Driver`) in the first column. 
