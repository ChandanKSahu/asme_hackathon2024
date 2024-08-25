import pandas as pd

req_395_loc = r"E:\OneDrive\OneDrive - Clemson University\Important\Thesis\VIPR GS\Codes\PyCh_Req_Analysis\INCOSE_RE\data\2021 - ConnectedVehiclePilotNYC_with_ReqID - 395.txt"
req_trace_loc = r"E:\OneDrive\OneDrive - Clemson University\Important\Thesis\VIPR GS\Codes\PyCh_Req_Analysis\INCOSE_RE\data\NYS-CVPD APPENDIX B Needs-to-Requirements Traceability Matrix (NRTM).txt"

req_395 = pd.read_csv(req_395_loc, sep="\t", index_col=None, header="infer")
req_trace_df = pd.read_csv(req_trace_loc, sep="\t", index_col=None, header="infer")

#%%
req_ids_395 = list(req_395.ReqID)
req_ids_in_trace = list(req_trace_df["ReqID"])

miss_req_ids = [k for k in req_ids_395 if k not in req_ids_in_trace]
#%%

import collections
print([item for item, count in collections.Counter(req_ids_395).items() if count > 1])