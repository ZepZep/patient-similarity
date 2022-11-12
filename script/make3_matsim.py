import os
import sys

AICOPE_PY_LIB = os.environ.get("AICOPE_PY_LIB")
if AICOPE_PY_LIB and AICOPE_PY_LIB not in sys.path: sys.path.append(AICOPE_PY_LIB)
from aicnlp.similarity import MatsimComputer

PACSIM_DATA = os.environ.get("AICOPE_SCRATCH") + "/pacsim"


relevant_patients = set([
    127, 388, 506, 584, 726, 780, 910, 913, 1023, 1061, 1088, 1157, 1213, 1219, 1244, 1268,
    1339, 1548, 1583, 1633, 1710, 1713, 1768, 1771, 1852, 1854, 1919, 1965, 1970, 2048, 2090,
    2104, 2124, 2179, 2405, 2457, 2515, 2746, 2757, 2770, 2941, 2960, 3098, 3099, 3106, 3119,
    3128, 3183, 3401, 3420, 3443, 3555, 3561, 3848, 3923, 4024, 4197, 4233
])

computer = MatsimComputer(PACSIM_DATA)

computer.calculate(
    # patterns=["*Vd2v*.feather"],
    methods=[
        ("Reds", {
            "patients": relevant_patients,
            "id_letter": "R",
        }),
    ],
)

# computer.calculate(
#     # patterns=["*Vd2v*.feather"],
#     methods=[
#         ("Rrv2", {
#             "patients": relevant_patients,
#             "id_letter": "R",
#         }),
#         ("Rmms", {
#             "patients": relevant_patients,
#             "id_letter": "R",
#         }),
#     ],
# )
