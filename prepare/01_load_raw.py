# read the raw BRFSS 2015 file and keep the columns we need
# raw file: https://www.cdc.gov/brfss/annual_data/2015/files/LLCP2015XPT.zip
import sys, pyreadstat
RAW = sys.argv[1] if len(sys.argv) > 1 else "LLCP2015.XPT"
KEEP = """DIABETE3 EXERANY2 EXRACT11 EXEROFT1 EXERHMM1 EXRACT21 STRENGTH _PACAT1 PA1MIN_ METVL11_ PAMIN11_ _TOTINDA
_RFHYPE5 TOLDHI2 _CHOLCHK _BMI5 SMOKE100 CVDSTRK3 _MICHD _FRTLT1 _VEGLT1 _RFDRHV5 HLTHPLN1 MEDCOST GENHLTH
MENTHLTH PHYSHLTH DIFFWALK SEX _AGEG5YR EDUCA INCOME2 _LLCPWT _STSTR _PSU""".split()
df, _ = pyreadstat.read_xport(RAW, usecols=KEEP)
df.to_csv("../data/brfss2015_selected.csv.gz", index=False)
print(df.shape)
