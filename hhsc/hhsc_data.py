# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:01:03 2019

@author: ankit.patel
"""

import glob
import re
import pandas as pd
import os
import numpy as np
from datetime import datetime
from functools import reduce

currentMonth = '0' + str(datetime.now().month)
currentYear = str(datetime.now().year)

today = pd.Timestamp('today').normalize()
end_date = today - pd.DateOffset(months=18)
end_date2 = today - pd.DateOffset(months=36)

ahca_file = max([os.path.join('I:\\ahca\\data\\raw_data\\pro30',d) for d in os.listdir('I:\\ahca\\data\\raw_data\\pro30')], key=os.path.getmtime)

mick_directory = 'I:\\public_data\\data\\raw_data\\curated\\'
all_subdirs = [d for d in os.listdir(mick_directory) if os.path.isdir(os.path.join(mick_directory,d))]
#mick_directory = 'J:\\analytics\\110_hh_scorecard\\data\\raw_data\\'
mick_recent_dir = max([os.path.join(mick_directory,d) for d in all_subdirs], key=os.path.getmtime)

pub_directory = 'I:\\home_health\\data\\raw_data\\'
pub_recent_dir = max([os.path.join(pub_directory,d) for d in os.listdir(pub_directory)], key=os.path.getmtime)

oscar_directory = 'I:\\public_data\\data\\derived_data\\monthly'
oscar_recent_dir = max([os.path.join(oscar_directory,d) for d in os.listdir(oscar_directory)], key=os.path.getmtime)

county_file = pd.read_excel('I:\\geographic_variation\\data\\raw_data\\zipe_codes_states.xlsx', usecols=['city','county','zip_code'])


# npi_file = pd.read_csv('I:\\npi\\data\\raw_data\\npidata_20050523-20180107.csv',
#                        iterator=True, chunksize=10000, usecols=['NPI', 'Entity Type Code', 'Healthcare Provider Taxonomy Code_1',
#                                                                'Provider Business Mailing Address State Name','Provider Business Practice Location Address City Name',
#                                                                'Provider Enumeration Date','Provider First Line Business Mailing Address'])

# npi_df = pd.concat([chunk[(chunk['Healthcare Provider Taxonomy Code_1'] == '251E00000X') & (chunk['Entity Type Code'] == 2)] for chunk in npi_file])

# npi_df = npi_df.join(npi_df['Provider First Line Business Mailing Address'].str.upper().str.replace('[^\w\s]','').str.split(' ',expand=True).add_prefix('street'))

orig_list = ['BOULEVARD', 'CIR', 'DR', 'ST', 'AVE', 'N', 'S', 'E', 'W', 'FIRST', 'SECOND',
             'THIRD', 'FOURTH', 'FIFTH', 'SIXTH', 'SEVENTH', 'EIGHTH', 'NINTH']

rep_list = ['BLVD', 'CIRCLE', 'DRIVE', 'STREET','AVENUE', 'NORTH', 'SOUTH',
            'EAST', 'WEST', '1ST', '2ND', '3RD', '4TH', '5TH', '6TH', '7TH', '8TH', '9TH']



county_file['zip_code_len'] = county_file.zip_code.map(str).apply(len)

county_file['zip_code'] = np.where(county_file['zip_code_len'] == 3, '00' + county_file['zip_code'].astype(str), county_file['zip_code'].astype(str))
county_file['zip_code'] = np.where(county_file['zip_code_len'] == 4, '0' + county_file['zip_code'].astype(str), county_file['zip_code'].astype(str))
# def tidy_split(df, column, sep='|', keep=False):
#     indexes = list()
#     new_values = list()
#     df = df.dropna(subset=[column])
#     for i, presplit in enumerate(df[column].astype(str)):
#         values = presplit.split(sep)
#         if keep and len(values) > 1:
#             indexes.append(i)
#             new_values.append(presplit)
#         for value in values:
#             indexes.append(i)
#             new_values.append(value)
#     new_df = df.iloc[indexes, :].copy()
#     new_df[column] = new_values
#     return new_df

# county_file = tidy_split(county_file, 'zips', sep = ' ')

ahca_data = pd.read_sas(ahca_file,format='sas7bdat')
oscar_data = pd.read_sas(oscar_recent_dir + '\oscar.sas7bdat', format='sas7bdat')

ahca_data["CCN"] = ahca_data['CCN'].str.decode("utf-8")

oscar_str_df = oscar_data.select_dtypes([np.object])
oscar_str_df = oscar_str_df.stack().str.decode("utf-8").unstack()

for col in oscar_str_df:
    oscar_data[col] = oscar_str_df[col]

state_snf_rehosp = pd.merge(ahca_data[['CCN','rate_adj']],
                            oscar_data[['PROV_NUM','Survey_date','STATE_ABBREV','ZIP_CD']]
                            , left_on='CCN', right_on='PROV_NUM', how='left')

state_snf_rehosp = state_snf_rehosp.sort_values(['CCN','Survey_date']).groupby(['CCN']).tail(1)

state_snf_state_avg = state_snf_rehosp.groupby(['STATE_ABBREV'])[['rate_adj']].mean().reset_index()

mick_files = [f.split('.')[0] for f in glob.glob(mick_recent_dir + "\\HHA\*.csv") if os.path.isfile(os.path.join(mick_recent_dir, f))]
mick_files = [x for x in mick_files if re.findall("certification|health_defs|compdet|compsrv",x)]

pub_files = [f.split('.')[0] for f in glob.glob(pub_recent_dir + "\*.csv") if os.path.isfile(os.path.join(pub_recent_dir, f))]

dict_pub={}
for file in pub_files:
    file_name = file.split("\\")[-1]
    dict_pub[file_name] = pd.read_csv(file+'.csv')
    for key in dict_pub.keys():
        [df.rename(columns = lambda x: x.strip().lower().replace(' ','_').replace('(','').replace(')','').replace(',',''), inplace=True) for df in dict_pub.values()]
        if key == 'HHC_SOCRATA_MSR_DT_RNG':
            dt_rng = dict_pub['HHC_SOCRATA_MSR_DT_RNG']
            dt_rng['measure_name'] = dt_rng['measure_name'].str.replace(',',';')
            dt_rng['measure_date_range'] = dt_rng['measure_date_range'].str.replace(',',';')
            dt_rng = dt_rng[~dt_rng['measure_name'].str.contains('|'.join(['Average','31 days','medication issues','ulcers']))]
            dict_pub['HHC_SOCRATA_MSR_DT_RNG'] = dt_rng
        if key == 'HHC_SOCRATA_HHCAHPS_PRVDR':
            hh_prvdr = dict_pub['HHC_SOCRATA_HHCAHPS_PRVDR']
            hh_prvdr['cms_certification_number_ccn'] = hh_prvdr['cms_certification_number_ccn'].astype(str)
            hh_prvdr['zip'] = hh_prvdr['zip'].apply('{:0>5}'.format)
            hh_prvdr['cms_certification_number_ccn'] = hh_prvdr['cms_certification_number_ccn'].apply('{:0>6}'.format)
            dict_pub['HHC_SOCRATA_HHCAHPS_PRVDR'] = hh_prvdr



dict_mick={}
for file in mick_files:
    file_name = file.split("\\")[-1]
    dict_mick[file_name] = pd.read_csv(file +'.csv')
    mick_recent_date = mick_recent_dir.split("\\")[-1]
    for key in dict_mick.keys():
        if key == 'qies_health_defs_' + mick_recent_date:
            dict_mick['qies_health_defs_' + mick_recent_date].rename(columns={'TAG_NUM': 'DFCNCY_TAG_NUM', 'CLNDR_SRVY_DT_SK': 'SRVY_DT',
                     'PREX_CD': 'DFCNCY_PREX_CD', 'PRVDR_CCN': 'PRVDR_NUM'}, inplace=True)
            comp_surv_df = dict_mick['qies_health_defs_' + mick_recent_date][['PRVDR_NUM','SRVY_DT','SRVY_CYC_CD','DFCNCY_TAG_NUM','DFCNCY_PREX_CD']]
            comp_surv_df['PRVDR_NUM'] = comp_surv_df['PRVDR_NUM'].apply('{:0>6}'.format)
            standard_last_survey = comp_surv_df.sort_values(['PRVDR_NUM','SRVY_DT','DFCNCY_TAG_NUM']).groupby(['PRVDR_NUM']).tail(1)
            standard_last_survey = pd.merge(standard_last_survey[['PRVDR_NUM','SRVY_DT']],
                                             comp_surv_df,
                                             on=['PRVDR_NUM','SRVY_DT'],
                                             how='inner')

        if key == 'compsrv_' + mick_recent_date:
            dict_mick['compsrv_' + mick_recent_date].rename(columns={'TAG_NUM_15': 'DFCNCY_TAG_NUM', 'CLNDR_SRVY_DT_SK_10': 'SRVY_DT',
                     'SRVY_CYC_CD_4': 'SRVY_CYC_CD', 'PREX_CD_16': 'DFCNCY_PREX_CD', 'PRVDR_CCN': 'PRVDR_NUM'}, inplace=True)
            standard_complaint = dict_mick['compsrv_' + mick_recent_date][['PRVDR_NUM','SRVY_DT','SRVY_CYC_CD','DFCNCY_TAG_NUM','DFCNCY_PREX_CD']]
            standard_comp_survey = standard_complaint.sort_values(['PRVDR_NUM','SRVY_DT','DFCNCY_TAG_NUM']).groupby(['PRVDR_NUM']).tail(1)
            standard_comp_survey = pd.merge(standard_comp_survey[['PRVDR_NUM','SRVY_DT']],
                                            standard_complaint,
                                             on=['PRVDR_NUM','SRVY_DT'],
                                             how='inner')


standard_complaint_comb = pd.concat([standard_last_survey,standard_comp_survey])

standard_complaint_comb['SRVY_DT'] = standard_complaint_comb.SRVY_DT.str.replace('-','')

standard_complaint_comb['DFCNCY_TAG_NUM'] = standard_complaint_comb['DFCNCY_TAG_NUM'].fillna(0).astype(int).astype(str)
standard_complaint_comb['TagLen'] = standard_complaint_comb['DFCNCY_TAG_NUM'].map(len)
standard_complaint_comb['SRVY_DT'] = pd.to_datetime(standard_complaint_comb['SRVY_DT'].astype(str), format='%Y%m%d')

def testtag(row):
    if row['TagLen'] == 1 and row['DFCNCY_TAG_NUM'] != '0':
        return str(row['DFCNCY_PREX_CD']) + '-000' + str(row['DFCNCY_TAG_NUM'])
    elif row['TagLen'] == 2:
        return str(row['DFCNCY_PREX_CD']) + '-00' + str(row['DFCNCY_TAG_NUM'])
    elif row['TagLen'] == 3:
        return str(row['DFCNCY_PREX_CD']) + '-0' + str(row['DFCNCY_TAG_NUM'])
    elif row['TagLen'] == 4:
        return str(row['DFCNCY_PREX_CD']) + '-' + str(row['DFCNCY_TAG_NUM'])

standard_complaint_comb['DFCNCY_TAG_NUM'] = standard_complaint_comb.apply(testtag,axis=1)

gtag_combine = pd.merge(standard_complaint_comb,
                        dict_pub['HHC_SOCRATA_HHCAHPS_PRVDR'][['provider_name','address','cms_certification_number_ccn','state','zip']],
                        left_on='PRVDR_NUM',right_on='cms_certification_number_ccn',
                        how='inner')

gtag_combine['zip'] = gtag_combine['zip'].apply('{:0>5}'.format)

gtag_full_list = pd.merge(gtag_combine,
                          county_file[['county','zip_code']],
                          left_on='zip', right_on='zip_code',
                          how='inner')

gtag_full_list = gtag_full_list.drop_duplicates(subset=['PRVDR_NUM','SRVY_DT','DFCNCY_TAG_NUM']).dropna(subset=['DFCNCY_TAG_NUM'])

mask_date_standard = (gtag_full_list['SRVY_DT'] > end_date2) & (gtag_full_list['SRVY_DT'] <= today) & (gtag_full_list['SRVY_CYC_CD'] == 'S')
mask_date_complaint = (gtag_full_list['SRVY_DT'] > end_date) & (gtag_full_list['SRVY_DT'] <= today) & (gtag_full_list['SRVY_CYC_CD'] == 'C')

gtag_standard = gtag_full_list.loc[mask_date_standard]
gtag_complaint = gtag_full_list.loc[mask_date_complaint]

gtag_standard['tagcount_lastsurv'] = gtag_standard.groupby(["PRVDR_NUM","SRVY_DT"])["DFCNCY_TAG_NUM"].transform('count')
gtag_complaint['tagcount_complaint'] = gtag_complaint.groupby(["PRVDR_NUM","SRVY_DT"])["DFCNCY_TAG_NUM"].transform('count')

gtag_standard_complaint = pd.merge(gtag_standard,
                                   gtag_complaint[['PRVDR_NUM','tagcount_complaint']],
                                   on='PRVDR_NUM',
                                   how='left')

gtag_standard_complaint = pd.concat([gtag_standard_complaint,gtag_complaint],sort=True).\
drop_duplicates(subset=['PRVDR_NUM','SRVY_DT', 'SRVY_CYC_CD','DFCNCY_TAG_NUM']).sort_values(by=['PRVDR_NUM','SRVY_DT','SRVY_CYC_CD','DFCNCY_TAG_NUM'])\
                        [['provider_name','address','PRVDR_NUM','state','zip','SRVY_DT','SRVY_CYC_CD','DFCNCY_TAG_NUM','tagcount_lastsurv','tagcount_complaint']].sort_values(by=['provider_name'])

gtag_standard_complaint['SRVY_CYC_CD'] = np.where(gtag_standard_complaint['SRVY_CYC_CD'] == 'S', 'Standard', 'Complaint')

output_directory = '\\\\worker1\\e\\analytics\\development\\monthly_production\\110_hh_scorecard\\outputs'
out_recent_dir = max([os.path.join(output_directory,d) for d in os.listdir(output_directory)], key=os.path.getmtime)

for state in gtag_standard_complaint['state'].unique():
  if any(x in state for x in ('AZ', 'CA', 'FL', 'NV', 'MD')):
    file_name = '\\Gtag_Info_{0}.csv'.format(state)
    gtag_standard_complaint[gtag_standard_complaint['state'] == state].to_csv(out_recent_dir + file_name,index=False)

gtag_standard_complaint['comp_binary'] = np.where(gtag_standard_complaint['tagcount_complaint'].notnull(),1,0)

gtag_standard_complaint['def_binary'] = np.where(gtag_standard_complaint["DFCNCY_TAG_NUM"].notnull(),1,0)
gtag_standard_complaint['def_total'] = gtag_standard_complaint.groupby(['PRVDR_NUM','SRVY_DT'])["DFCNCY_TAG_NUM"].transform('count')

patient_rights_df = gtag_standard_complaint.groupby(['PRVDR_NUM','SRVY_DT'])['DFCNCY_TAG_NUM']\
.apply(lambda x: x[x.isin(["G-0100","G-0101","G-0102","G-0103","G-0104",
					  "G-0105","G-0106","G-0107","G-0108","G-0109",
					  "G-0110","G-0111","G-0112","G-0113","G-0114",
					  "G-0115","G-0116"])].count()).reset_index(name='def_patient_rights')



compliance_df = gtag_standard_complaint.groupby(['PRVDR_NUM','SRVY_DT'])['DFCNCY_TAG_NUM']\
.apply(lambda x: x[x.isin(["G-0117","G-0118","G-0119","G-0120","G-0121"])].count()).reset_index(name='def_compliance')

org_admin_df = gtag_standard_complaint.groupby(['PRVDR_NUM','SRVY_DT'])['DFCNCY_TAG_NUM']\
.apply(lambda x: x[x.isin(["G-0122","G-0123","G-0124","G-0125","G-0126",
					  "G-0127","G-0128","G-0129","G-0130","G-0131",
					  "G-0132","G-0133","G-0134","G-0135","G-0136",
					  "G-0137","G-0138","G-0139","G-0140","G-0141",
					  "G-0142","G-0143","G-0144","G-0145","G-0146",
					  "G-0147","G-0148","G-0149","G-0150"])].count()).reset_index(name='def_org_admin')

prof_personel = gtag_standard_complaint.groupby(['PRVDR_NUM','SRVY_DT'])['DFCNCY_TAG_NUM']\
.apply(lambda x: x[x.isin(["G-0151","G-0152","G-0153","G-0154","G-0155"])].count()).reset_index(name='def_prof_personel')

med_poc = gtag_standard_complaint.groupby(['PRVDR_NUM','SRVY_DT'])['DFCNCY_TAG_NUM']\
.apply(lambda x: x[x.isin(["G-0156","G-0157","G-0158","G-0159","G-0160",
					  "G-0161","G-0162","G-0163","G-0164","G-0165",
					  "G-0166","G-0300"])].count()).reset_index(name='def_med_poc')

report_info = gtag_standard_complaint.groupby(['PRVDR_NUM','SRVY_DT'])['DFCNCY_TAG_NUM']\
.apply(lambda x: x[x.isin(["G-0320","G-0321","G-0322","G-0324","G-0325",
					  "G-0326","G-0327","G-0328"])].count()).reset_index(name='def_report_info')

skilled_nurse = gtag_standard_complaint.groupby(['PRVDR_NUM','SRVY_DT'])['DFCNCY_TAG_NUM']\
.apply(lambda x: x[x.isin(["G-0168","G-0169","G-0170","G-0171","G-0172",
					  "G-0173","G-0174","G-0175","G-0176","G-0177",
					  "G-0178","G-0179","G-0180","G-0181","G-0182",
					  "G-0183"])].count()).reset_index(name='def_skilled_nurse')

therapy_serv = gtag_standard_complaint.groupby(['PRVDR_NUM','SRVY_DT'])['DFCNCY_TAG_NUM']\
.apply(lambda x: x[x.isin(["G-0184","G-0185","G-0186","G-0187","G-0188",
					  "G-0189","G-0190","G-0191","G-0192","G-0193"])].count()).reset_index(name='def_therapy_service')

social_serv = gtag_standard_complaint.groupby(['PRVDR_NUM','SRVY_DT'])['DFCNCY_TAG_NUM']\
.apply(lambda x: x[x.isin(["G-0194","G-0195","G-0196","G-0197","G-0198",
					  "G-0199","G-0200","G-0201"])].count()).reset_index(name='def_social_service')


hha_service = gtag_standard_complaint.groupby(['PRVDR_NUM','SRVY_DT'])['DFCNCY_TAG_NUM']\
.apply(lambda x: x[x.isin(["G-0202","G-0203","G-0204","G-0205","G-0206",
					  "G-0207","G-0208","G-0209","G-0210","G-0211",
					  "G-0212","G-0213","G-0214","G-0215","G-0216",
					  "G-0217","G-0218","G-0219","G-0220","G-0221",
					  "G-0222","G-0223","G-0224","G-0225","G-0226",
					  "G-0227","G-0228","G-0229","G-0230","G-0231",
					  "G-0232","G-0233","G-0301","G-0302"])].count()).reset_index(name='def_hha_service')

clinical_recs = gtag_standard_complaint.groupby(['PRVDR_NUM','SRVY_DT'])['DFCNCY_TAG_NUM']\
.apply(lambda x: x[x.isin(["G-0235","G-0236","G-0237","G-0238","G-0239",
					  "G-0240","G-0241","G-0303"])].count()).reset_index(name='def_clinical_records')

program_eval = gtag_standard_complaint.groupby(['PRVDR_NUM','SRVY_DT'])['DFCNCY_TAG_NUM']\
.apply(lambda x: x[x.isin(["G-0242","G-0243","G-0244","G-0245","G-0246",
					  "G-0247","G-0248","G-0249","G-0250","G-0251"])].count()).reset_index(name='def_program_eval')

comp_assessment = gtag_standard_complaint.groupby(['PRVDR_NUM','SRVY_DT'])['DFCNCY_TAG_NUM']\
.apply(lambda x: x[x.isin(["G-0330","G-0331","G-0332","G-0333","G-0334",
					  "G-0335","G-0336","G-0337","G-0338","G-0339",
					  "G-0340","G-0341","G-0342"])].count()).reset_index(name='def_comp_assess')

dfs = [gtag_standard_complaint, patient_rights_df, compliance_df, org_admin_df, prof_personel,
       med_poc, report_info, skilled_nurse, therapy_serv, social_serv, hha_service, clinical_recs,
       program_eval, comp_assessment]

dfs = [df.set_index(['PRVDR_NUM', 'SRVY_DT']) for df in dfs]
complaint_cols_join = dfs[0].join(dfs[1:]).sort_values(['PRVDR_NUM','SRVY_DT','DFCNCY_TAG_NUM']).groupby(['PRVDR_NUM']).head(1)


dict_mick['compdet_' + mick_recent_date].rename(columns={'CLNDR_EXIT_DT_SK': 'EXIT_DT', 'PRVDR_CCN': 'PRVDR_NUM'}, inplace=True)

dict_mick['compdet_' + mick_recent_date][['EXIT_DT']] = dict_mick['compdet_' + mick_recent_date].EXIT_DT.str.replace('-','')

dict_mick['compdet_' + mick_recent_date][['EXIT_DT']] = pd.to_datetime(dict_mick['compdet_' + mick_recent_date].EXIT_DT.astype(str), format='%Y%m%d')
alleg_mask = (dict_mick['compdet_' + mick_recent_date].EXIT_DT > end_date) & (dict_mick['compdet_' + mick_recent_date].EXIT_DT <= today)
allegations = dict_mick['compdet_' + mick_recent_date].loc[alleg_mask].drop_duplicates(subset=['PRVDR_NUM','EXIT_DT','ALGTN_TYPE_CD']).dropna(subset=['ALGTN_TYPE_CD'])\
[['PRVDR_NUM','EXIT_DT','ALGTN_TYPE_CD','ALGTN_FNDNG_CD']]

allegations['alleg_total'] = allegations.groupby(['PRVDR_NUM','EXIT_DT'])["ALGTN_FNDNG_CD"].transform('count')

allegations['sub_alleg_bin'] = np.where(allegations['ALGTN_FNDNG_CD']==1,1,0)

total_sub_alleg= allegations.assign(total_sub_alleg = lambda x:x['ALGTN_FNDNG_CD']==1)\
.groupby(['PRVDR_NUM','EXIT_DT'])['total_sub_alleg'].count().reset_index()

total_qoc_alleg= allegations.assign(total_qoc_alleg = lambda x:x['ALGTN_TYPE_CD']==11)\
.groupby(['PRVDR_NUM','EXIT_DT'])['total_qoc_alleg'].count().reset_index()

allegations['qoc_bin'] = np.where(allegations['ALGTN_TYPE_CD']==11,1,0)

total_abu_neg_alleg= allegations.assign(total_abu_neg_alleg = lambda x:x['ALGTN_TYPE_CD'].isin([1,2,3]))\
.groupby(['PRVDR_NUM','EXIT_DT'])['total_abu_neg_alleg'].count().reset_index()

allegations['abu_neg_bin'] = np.where(allegations['ALGTN_TYPE_CD'].isin([1,2,3]),1,0)

total_staff_alleg= allegations.assign(total_staff_alleg = lambda x:x['ALGTN_TYPE_CD'].isin([15,16,17,18,19,30]))\
.groupby(['PRVDR_NUM','EXIT_DT'])['total_staff_alleg'].count().reset_index()

allegations['staff_bin'] = np.where(allegations['ALGTN_TYPE_CD'].isin([15,16,17,18,19,30]),1,0)

total_rehosp_alleg= allegations.assign(total_rehosp_alleg = lambda x:x['ALGTN_TYPE_CD'].isin([1,2,3,11,15,16,17,18,19,30]))\
.groupby(['PRVDR_NUM','EXIT_DT'])['total_rehosp_alleg'].count().reset_index()

allegations['rehosp_bin'] = np.where(allegations['ALGTN_TYPE_CD'].isin([1,2,3,11,15,16,17,18,19,30]),1,0)

def weight_alleg(x):
    if x == 1:
        return 2
    elif x == 2:
        return 1
    else:
        return 0

func = np.vectorize(weight_alleg)
weight_flag = func(allegations['ALGTN_FNDNG_CD'])
allegations['weight_flag'] = weight_flag

allegations['total_weight_alleg'] = allegations.groupby(['PRVDR_NUM','EXIT_DT','ALGTN_FNDNG_CD'])['weight_flag'].transform('sum')

dflist = [allegations, total_sub_alleg,total_rehosp_alleg,total_qoc_alleg,total_abu_neg_alleg,total_staff_alleg]
allegations = reduce(lambda left,right: pd.merge(left,right, on=['PRVDR_NUM','EXIT_DT'],
                                                 how='inner'), dflist)

alleg_last_recs = allegations.sort_values(['PRVDR_NUM','EXIT_DT']).groupby(['PRVDR_NUM']).tail(1)

dict_mick['qies_certification_' + mick_recent_date].rename(columns={'CLNDR_CRTFCTN_DT_SK': 'CRTFCTN_DT', 'PRVDR_CCN': 'PRVDR_NUM',
                                                                     'CLNDR_CRTFCTN_UPDT_DT_SK': 'CRTFCTN_UPDT_DT'}, inplace=True)

dict_mick['qies_certification_' + mick_recent_date][['CRTFCTN_DT']] = dict_mick['qies_certification_' + mick_recent_date].CRTFCTN_DT.str.replace('-','')


dict_mick['qies_certification_' + mick_recent_date][['CRTFCTN_DT']] = pd.to_datetime(dict_mick['qies_certification_' + mick_recent_date].CRTFCTN_DT.astype(str), format='%Y%m%d')
dict_mick['qies_certification_' + mick_recent_date][['CRTFCTN_UPDT_DT']] = dict_mick['qies_certification_' + mick_recent_date].CRTFCTN_UPDT_DT.astype(str)
standard_survey = dict_mick['qies_certification_' + mick_recent_date]
standard_survey['PRVDR_NUM'] = standard_survey['PRVDR_NUM'].apply('{:0>6}'.format)
standard_survey = standard_survey[standard_survey.columns.drop(list(standard_survey.filter(regex='NA')))]
standard_survey['lag_survey_date'] = standard_survey['CRTFCTN_DT'].shift(1)
standard_survey['lag_survey_date2'] = standard_survey['CRTFCTN_DT'].shift(2)
standard_survey['provider_lag'] = standard_survey['PRVDR_NUM'].shift(1)
standard_survey['provider_lag2'] = standard_survey['PRVDR_NUM'].shift(2)
standard_survey['survey_date_diff'] = (standard_survey['CRTFCTN_DT'] - standard_survey['lag_survey_date']).astype('timedelta64[D]')

standard_survey['lag_survey_date'] = np.where(standard_survey['PRVDR_NUM'] != standard_survey['provider_lag'], np.datetime64('NaT'),standard_survey['lag_survey_date'])
standard_survey['survey_date_diff'] = np.where(standard_survey['PRVDR_NUM'] != standard_survey['provider_lag'], np.nan,standard_survey['survey_date_diff'])
standard_survey['survey_year_diff'] = standard_survey['survey_date_diff'] / 365
standard_survey['survey_year_diff'] = np.where(standard_survey['PRVDR_NUM'] != standard_survey['provider_lag'], np.nan,standard_survey['survey_year_diff'])
standard_survey['survey_date_diff2'] = standard_survey['survey_date_diff']
standard_survey['survey_year_diff2'] = standard_survey['survey_year_diff']

standard_survey['survey_date_diff2'] = np.where((standard_survey['survey_date_diff'] == 1) |
               (standard_survey['survey_date_diff'].between(1,100) & standard_survey['CNSUS_ADMSN_UNDUPD_CNT'] > 1000),
               (standard_survey['CRTFCTN_DT'] - standard_survey['lag_survey_date2']).astype('timedelta64[D]'), np.nan)

standard_survey['survey_year_diff2'] = np.where((standard_survey['survey_date_diff'] == 1) |
               (standard_survey['survey_date_diff'].between(1,100) & standard_survey['CNSUS_ADMSN_UNDUPD_CNT'] > 1000),
               (standard_survey['survey_year_diff2'] / 365) , np.nan)

standard_survey['survey_date_diff2'] = np.where(standard_survey['PRVDR_NUM'] != standard_survey['provider_lag2'], np.nan,standard_survey['survey_date_diff2'])
standard_survey['survey_year_diff2'] = np.where(standard_survey['PRVDR_NUM'] != standard_survey['provider_lag2'], np.nan,standard_survey['survey_year_diff2'])

standard_survey['census_undup_admit_per_year'] = standard_survey['CNSUS_ADMSN_UNDUPD_CNT']/standard_survey['survey_year_diff2']
standard_survey['census_readmis_admit_per_year'] = standard_survey['CNSUS_ADMSN_READMSN_CNT']/standard_survey['survey_year_diff2']
standard_survey['census_discharge_hosp_per_year'] = standard_survey['CNSUS_HOSP_DSCHRG_CNT']/standard_survey['survey_year_diff2']
standard_survey['census_discharge_snf_per_year'] = standard_survey['NH_DSCHRG_CNT']/standard_survey['survey_year_diff2']
standard_survey['census_dischrg_goal_per_year'] = standard_survey['CNSUS_DSCHRG_GOAL_MET_CNT']/standard_survey['survey_year_diff2']
standard_survey['census_dischrg_death_per_year'] = standard_survey['CNSUS_DEATH_DSCHRG_CNT']/standard_survey['survey_year_diff2']
standard_survey['discharge_count_per_year'] = standard_survey['DSCHRG_CNT']/standard_survey['survey_year_diff2']

standard_survey['pred_all_small_flag'] = np.where(standard_survey['census_undup_admit_per_year'] < 10, 1, 0)
standard_survey['pred_med_other_small_flag'] = np.where(standard_survey['census_undup_admit_per_year'] < 15, 1, 0)
standard_survey['pred_surg_other_small_flag'] = np.where(standard_survey['census_undup_admit_per_year'] < 54, 1, 0)
standard_survey['pred_CHF_small_flag'] = np.where(standard_survey['census_undup_admit_per_year'] < 125, 1, 0)
standard_survey['pred_med_high_risk_small_flag'] = np.where(standard_survey['census_undup_admit_per_year'] < 33, 1, 0)
standard_survey['pred_med_low_risk_small_flag'] = np.where(standard_survey['census_undup_admit_per_year'] < 15, 1, 0)
standard_survey['pred_surg_high_risk_small_flag'] = np.where(standard_survey['census_undup_admit_per_year'] < 54, 1, 0)
standard_survey['pred_surg_low_risk_small_flag'] = np.where(standard_survey['census_undup_admit_per_year'] < 125, 1, 0)
standard_survey['pred_lung_small_flag'] = np.where(standard_survey['census_undup_admit_per_year'] < 83, 1, 0)

stand_survlast_recs = standard_survey.sort_values(['PRVDR_NUM','CRTFCTN_DT']).groupby(['PRVDR_NUM']).tail(1)
stand_survlast_recs['CRTFCTN_DT'] = stand_survlast_recs.CRTFCTN_DT.astype(str).str.replace('-','')

hhc_zip = dict_pub['HHC_SOCRATA_ZIP']

hhc_zip['zip_code'] = hhc_zip['zip_code'].astype(str)

county_freq = pd.merge(county_file,
                        hhc_zip,
                        on='zip_code',
                        how='inner')

county_freq['County_ID'] = county_freq['state'] + '_' + county_freq['county']

county_last_recs = county_freq.sort_values(['cms_certification_number_ccn']).groupby(['County_ID','cms_certification_number_ccn']).tail(1)

county_last_recs['County_count'] = county_last_recs.groupby('County_ID')['cms_certification_number_ccn'].transform('count')

county_last_recs['cms_certification_number_ccn'] = county_freq['cms_certification_number_ccn'].apply('{:0>6}'.format)


prvdr_socrata = dict_pub['HHC_SOCRATA_PRVDR']
qm_final = prvdr_socrata[prvdr_socrata.columns.drop(list(prvdr_socrata.filter(regex='footnote')))]
qm_final.loc[:,'city'] = qm_final['city'].str.title()
qm_final.loc[:,'cms_certification_number_ccn'] = qm_final['cms_certification_number_ccn'].apply('{:0>6}'.format)

prvdr_hcaps = dict_pub['HHC_SOCRATA_HHCAHPS_PRVDR']
hcaps_final = prvdr_hcaps[prvdr_hcaps.columns.drop(list(prvdr_hcaps.filter(regex='footnote')))]
hcaps_final.loc[:,'city'] = hcaps_final['city'].str.title()
hcaps_final.loc[:,'cms_certification_number_ccn'] = hcaps_final['cms_certification_number_ccn'].apply('{:0>6}'.format)

all_merge = county_last_recs.merge(stand_survlast_recs, left_on='cms_certification_number_ccn', right_on='PRVDR_NUM', how='left')\
            .merge(state_snf_state_avg,left_on='state',right_on='STATE_ABBREV',how='left')\
            .merge(qm_final,on=['cms_certification_number_ccn','state'], how='left')\
            .merge(hcaps_final,on=['cms_certification_number_ccn','state',
                                   'offers_nursing_care_services', 'offers_physical_therapy_services',
                                   'offers_occupational_therapy_services','offers_speech_pathology_services', 'offers_medical_social_services',
                                   'offers_home_health_aide_services'], how='left')\
            .merge(complaint_cols_join, left_on=['cms_certification_number_ccn','state']\
                   ,right_on=['PRVDR_NUM','state'], how='left')\
            .merge(alleg_last_recs, left_on='cms_certification_number_ccn', right_on='PRVDR_NUM', how='left')\


all_merge['alleg_binary'] = np.where(all_merge['alleg_total'] > 0, 1, 0)
all_merge['weighted_alleg_binary'] = all_merge['weight_flag']
all_merge['for_profit_binary'] = np.where(all_merge['GNRL_CNTL_TYPE_DESC'] == 'PROPRIETARY', 1, 0)
all_merge['no_lab_service'] = np.where(all_merge['LAB_SRVC_DESC'] == 'NOT PROVIDED', 1, 0)
all_merge['no_pharm_service'] = np.where(all_merge['PHRMCY_SRVC_DESC'] == 'NOT PROVIDED', 1, 0)
all_merge['no_other_service'] = np.where(all_merge['OTHR_SRVC_DESC'] == 'NOT PROVIDED', 1, 0)
all_merge['medicare_only_binary'] = np.where(all_merge['PGM_PRTCPTN_DESC'] == 'MEDICARE ONLY', 1, 0)
all_merge['medical_admin_binary'] = np.where(all_merge['DSCPLN_ADMINR_DESC'].isin(["RN/LPN","PHYSICIAN"]), 1, 0)
all_merge['staff_nursing_fulltime_binary'] = np.where(all_merge['NRSNG_SRVC_DESC'] == 'PROVIDED BY STAFF', 1, 0)
all_merge['staff_pt_fulltime_binary'] = np.where(all_merge['PT_SRVC_DESC'] == 'PROVIDED BY STAFF', 1, 0)
all_merge['staff_ot_fulltime_binary'] = np.where(all_merge['OCPTNL_THRPST_SRVC_DESC'] == 'PROVIDED BY STAFF', 1, 0)
all_merge['staff_speech_fulltime_binary'] = np.where(all_merge['SPCH_THRPY_SRVC_DESC'] == 'PROVIDED BY STAFF', 1, 0)
all_merge['staff_medsoc_fulltime_binary'] = np.where(all_merge['MDCL_SCL_SRVC_DESC'] == 'PROVIDED BY STAFF', 1, 0)
all_merge['staff_hhaide_fulltime_binary'] = np.where(all_merge['HH_AIDE_SRVC_DESC'] == 'PROVIDED BY STAFF', 1, 0)
all_merge['no_hha_serv_binary'] = np.where(all_merge['HH_AIDE_SRVC_DESC'] == 'NOT PROVIDED', 1, 0)
all_merge['staff_nutrit_fulltime_binary'] = np.where(all_merge['NTRTNL_GDNC_SRVC_DESC'] == 'NOT PROVIDED', 1, 0)
all_merge['survey_good_care_binary'] = np.where(all_merge['SRVYR_SMRY_DESC'] == 'CARE THAT PROMOTES POTENTIAL', 1, 0)
all_merge['rn_over_tot_licensed'] = all_merge['RN_CNT'] / (all_merge['RN_CNT'] + all_merge['LPN_LVN_CNT'])
all_merge['aide_over_tot_licensed'] = all_merge['HHA_AIDE_CNT'] / (all_merge['RN_CNT'] + all_merge['LPN_LVN_CNT'])
all_merge['tot_licensed_per_100_undup_admit'] = (all_merge['RN_CNT'] + all_merge['LPN_LVN_CNT'])*100 / all_merge['census_undup_admit_per_year']
all_merge['rn_per_100_undup_admit'] = (all_merge['RN_CNT'] / all_merge['census_undup_admit_per_year'])*100

all_merge['qm_timely_manner_low_binary'] = np.where(all_merge['how_often_the_home_health_team_began_their_patients’_care_in_a_timely_manner'] < 80, 1, 0)
all_merge['qm_timely_manner_high_binary'] = np.where(all_merge['how_often_the_home_health_team_began_their_patients’_care_in_a_timely_manner'] > 97, 1, 0)
all_merge['qm_risk_fall_low_binary'] = np.where(all_merge['how_often_the_home_health_team_checked_patients’_risk_of_falling'] < 90, 1, 0)
all_merge['qm_risk_fall_high_binary'] = np.where(all_merge['how_often_the_home_health_team_checked_patients’_risk_of_falling'] > 99, 1, 0)
all_merge['qm_check_depression_low_binary'] = np.where(all_merge['how_often_the_home_health_team_checked_patients_for_depression'] < 85, 1, 0)
all_merge['qm_check_depression_high_binary'] = np.where(all_merge['how_often_the_home_health_team_checked_patients_for_depression'] > 95, 1, 0)
all_merge['qm_immun_shots'] = (all_merge['how_often_the_home_health_team_determined_whether_patients_received_a_flu_shot_for_the_currnet_flu_season']\
                               + all_merge['how_often_the_home_health_team_made_sure_that_their_patients_have_received_a_pneumococcal_vaccine_pneumonia_shot']) / 2
all_merge['qm_immun_shots_low_binary'] = np.where(all_merge['qm_immun_shots'] < 60, 1, 0)
all_merge['qm_immun_shots_high_binary'] = np.where(all_merge['qm_immun_shots'] > 80, 1, 0)
all_merge['qm_diab_care_low_binary'] = np.where(all_merge['with_diabetes_how_often_the_home_health_team_got_doctor’s_orders_gave_foot_care_and_taught_patients_about_foot_care'] < 80, 1, 0)
all_merge['qm_diab_care_high_binary'] = np.where(all_merge['with_diabetes_how_often_the_home_health_team_got_doctor’s_orders_gave_foot_care_and_taught_patients_about_foot_care'] >= 98, 1, 0)
all_merge['qm_diab_care_med_binary'] = np.where(all_merge['with_diabetes_how_often_the_home_health_team_got_doctor’s_orders_gave_foot_care_and_taught_patients_about_foot_care']\
         .between(80,97), 1, 0)
all_merge['qm_func_status_avg'] = (all_merge['how_often_patients_got_better_at_walking_or_moving_around']\
                               + all_merge['how_often_patients_got_better_at_getting_in_and_out_of_bed']\
                               + all_merge['how_often_patients_got_better_at_bathing']\
                               + all_merge['how_often_patients_had_less_pain_when_moving_around']) / 4
all_merge['qm_func_status_avg_low_binary'] = np.where(all_merge['qm_func_status_avg'] < 30, 1, 0)
all_merge['qm_func_status_avg_high_binary'] = np.where(all_merge['qm_func_status_avg'] > 75, 1, 0)
all_merge['qm_func_status_avg_med_binary'] = np.where(all_merge['qm_func_status_avg']\
         .between(30,50), 1, 0)
all_merge['qm_improv_breathing_low_binary'] = np.where(all_merge['how_often_patients’_breathing_improved'] < 50, 1, 0)
all_merge['qm_improv_breathing_high_binary'] = np.where(all_merge['how_often_patients’_breathing_improved'] > 75, 1, 0)
all_merge['qm_improv_wounds_low_binary'] = np.where(all_merge['how_often_patients’_wounds_improved_or_healed_after_an_operation'] < 85, 1, 0)
all_merge['qm_improv_wounds_high_binary'] = np.where(all_merge['how_often_patients’_wounds_improved_or_healed_after_an_operation'] > 95, 1, 0)
all_merge['qm_correct_meds_low_binary'] = np.where(all_merge['how_often_patients_got_better_at_taking_their_drugs_correctly_by_mouth'] < 35, 1, 0)
all_merge['qm_correct_meds_high_binary'] = np.where(all_merge['how_often_patients_got_better_at_taking_their_drugs_correctly_by_mouth'] > 80, 1, 0)
all_merge['qm_correct_meds_med_binary'] = np.where(all_merge['how_often_patients_got_better_at_taking_their_drugs_correctly_by_mouth']\
         .between(65,80), 1, 0)
all_merge['qm_urgent_care_low_binary'] = np.where(all_merge['how_often_patients_receiving_home_health_care_needed_urgent_unplanned_care_in_the_er_without_being_admitted'] < 5, 1, 0)
all_merge['qm_urgent_care_high_binary'] = np.where(all_merge['how_often_patients_receiving_home_health_care_needed_urgent_unplanned_care_in_the_er_without_being_admitted'] > 15, 1, 0)

all_merge['cen_readmit_over_cen_undup'] = all_merge['CNSUS_ADMSN_READMSN_CNT'] / all_merge['CNSUS_ADMSN_UNDUPD_CNT']
all_merge['cen_nh_over_cen_nh_hosp'] = all_merge['NH_DSCHRG_CNT'] / (all_merge['NH_DSCHRG_CNT'] + all_merge['CNSUS_HOSP_DSCHRG_CNT'])
all_merge['cen_death_over_cen_undup'] = all_merge['CNSUS_DEATH_DSCHRG_CNT'] / all_merge['CNSUS_ADMSN_UNDUPD_CNT']
all_merge['cen_goal_met_over_cen_undup'] = all_merge['CNSUS_DSCHRG_GOAL_MET_CNT'] / all_merge['CNSUS_ADMSN_UNDUPD_CNT']

all_merge['def_patient_rights_bin'] = np.where(all_merge['def_patient_rights'] > 0, 1, 0)
all_merge['def_compliance_bin'] = np.where(all_merge['def_compliance'] > 0, 1, 0)
all_merge['def_org_admin_bin'] = np.where(all_merge['def_org_admin'] > 0, 1, 0)
all_merge['def_prof_personel_bin'] = np.where(all_merge['def_prof_personel'] > 0, 1, 0)
all_merge['def_med_poc_bin'] = np.where(all_merge['def_med_poc'] > 0, 1, 0)
all_merge['def_report_info_bin'] = np.where(all_merge['def_report_info'] > 0, 1, 0)
all_merge['def_skilled_nurse_bin'] = np.where(all_merge['def_skilled_nurse'] > 0, 1, 0)
all_merge['def_therapy_service_bin'] = np.where(all_merge['def_therapy_service'] > 0, 1, 0)
all_merge['def_social_service_bin'] = np.where(all_merge['def_social_service'] > 0, 1, 0)
all_merge['def_hha_service_bin'] = np.where(all_merge['def_hha_service'] > 0, 1, 0)
all_merge['def_clinical_records_bin'] = np.where(all_merge['def_clinical_records'] > 0, 1, 0)
all_merge['def_program_eval_bin'] = np.where(all_merge['def_program_eval'] > 0, 1, 0)

hcaps_cols = ['percent_of_patients_who_reported_that_their_home_health_team_discussed_medicines_pain_and_home_safety_with_them',\
          'percent_of_patients_who_reported_that_their_home_health_team_gave_care_in_a_professional_way',\
          'percent_of_patients_who_reported_that_their_home_health_team_communicated_well_with_them',\
          'percent_of_patients_who_gave_their_home_health_agency_a_rating_of_9_or_10_on_a_scale_from_0_lowest_to_10_highest',\
          'percent_of_patients_who_reported_yes_they_would_definitely_recommend_the_home_health_agency_to_friends_and_family']


all_merge[hcaps_cols] = all_merge[hcaps_cols].apply(pd.to_numeric,errors='coerce')

all_merge['HC_prof_care_low_binary'] = np.where(all_merge['percent_of_patients_who_reported_that_their_home_health_team_gave_care_in_a_professional_way'] < 82, 1, 0)
all_merge['HC_prof_care_high_binary'] = np.where(all_merge['percent_of_patients_who_reported_that_their_home_health_team_gave_care_in_a_professional_way'] > 94, 1, 0)

all_merge['HC_good_comm_low_binary'] = np.where(all_merge['percent_of_patients_who_reported_that_their_home_health_team_communicated_well_with_them'] < 78, 1, 0)
all_merge['HC_good_comm_high_binary'] = np.where(all_merge['percent_of_patients_who_reported_that_their_home_health_team_communicated_well_with_them'] > 92, 1, 0)

all_merge['HC_dis_med_pain_saf_low_binary'] = np.where(all_merge['percent_of_patients_who_reported_that_their_home_health_team_discussed_medicines_pain_and_home_safety_with_them'] < 75, 1, 0)
all_merge['HC_dis_med_pain_saf_high_binary'] = np.where(all_merge['percent_of_patients_who_reported_that_their_home_health_team_discussed_medicines_pain_and_home_safety_with_them'] > 91, 1, 0)

all_merge['HC_hh_rate_over8_low_binary'] = np.where(all_merge['percent_of_patients_who_gave_their_home_health_agency_a_rating_of_9_or_10_on_a_scale_from_0_lowest_to_10_highest'] < 75, 1, 0)
all_merge['HC_hh_rate_over8_high_binary'] = np.where(all_merge['percent_of_patients_who_gave_their_home_health_agency_a_rating_of_9_or_10_on_a_scale_from_0_lowest_to_10_highest'] > 93, 1, 0)

all_merge['HC_def_rec_low_binary'] = np.where(all_merge['percent_of_patients_who_reported_yes_they_would_definitely_recommend_the_home_health_agency_to_friends_and_family'] < 68, 1, 0)
all_merge['HC_def_rec_high_binary'] = np.where(all_merge['percent_of_patients_who_reported_yes_they_would_definitely_recommend_the_home_health_agency_to_friends_and_family'] > 89, 1, 0)

betas_all_worst = np.array([0.0525, 0.4924 ,-0.0810, 11.3188])

df_all_worst = all_merge[['how_often_home_health_patients_had_to_be_admitted_to_the_hospital', 'def_prof_personel',\
                          'percent_of_patients_who_reported_that_their_home_health_team_discussed_medicines_pain_and_home_safety_with_them',\
                          'rate_adj']]

all_merge['logit_all_worst'] = np.dot(betas_all_worst,df_all_worst.T) + 2.4127

all_merge['pred_all_worst'] = 1/(1+np.exp(-all_merge['logit_all_worst']))

betas_all_best = np.array([-0.0709, -0.0734 ,-0.1720, 0.5896, 0.5068, -7.4803])

df_all_best = all_merge[['how_often_home_health_patients_had_to_be_admitted_to_the_hospital', 'def_total',\
                          'weight_flag','qm_immun_shots_high_binary', 'qm_diab_care_high_binary','rate_adj']]

all_merge['logit_all_best'] = np.dot(betas_all_best,df_all_best.T) + 1.0807

all_merge['pred_all_best'] = 1/(1+np.exp(-all_merge['logit_all_best']))

all_merge['pred_all_diff'] = all_merge['pred_all_best'] - all_merge['pred_all_worst']

all_merge['oe_all_ptile'] = 0
all_merge.loc[all_merge['pred_all_diff'] <= -0.2767,'oe_all_ptile'] = 5
all_merge.loc[(all_merge['pred_all_diff'] > -0.2767) & (all_merge['pred_all_diff'] <= -0.2037), 'oe_all_ptile'] = 10
all_merge.loc[(all_merge['pred_all_diff'] > -0.2037) & (all_merge['pred_all_diff'] <= -0.1556),'oe_all_ptile'] = 15
all_merge.loc[(all_merge['pred_all_diff'] > -0.1556) & (all_merge['pred_all_diff'] <= -0.1325),'oe_all_ptile'] = 20
all_merge.loc[(all_merge['pred_all_diff'] > -0.1325) & (all_merge['pred_all_diff'] <= -0.1126),'oe_all_ptile'] = 25
all_merge.loc[(all_merge['pred_all_diff'] > -0.1126) & (all_merge['pred_all_diff'] <= -0.0843),'oe_all_ptile'] = 30
all_merge.loc[(all_merge['pred_all_diff'] > -0.0843) & (all_merge['pred_all_diff'] <= -0.0669),'oe_all_ptile'] = 35
all_merge.loc[(all_merge['pred_all_diff'] > -0.0669) & (all_merge['pred_all_diff'] <= -0.0427),'oe_all_ptile'] = 40
all_merge.loc[(all_merge['pred_all_diff'] > -0.0427) & (all_merge['pred_all_diff'] <= -0.0277),'oe_all_ptile'] = 45
all_merge.loc[(all_merge['pred_all_diff'] > -0.0277) & (all_merge['pred_all_diff'] <= -0.0104),'oe_all_ptile'] = 50
all_merge.loc[(all_merge['pred_all_diff'] > -0.0104) & (all_merge['pred_all_diff'] <= 0.0149),'oe_all_ptile'] = 55
all_merge.loc[(all_merge['pred_all_diff'] > 0.0149) & (all_merge['pred_all_diff'] <= 0.0351),'oe_all_ptile'] = 60
all_merge.loc[(all_merge['pred_all_diff'] > 0.0351) & (all_merge['pred_all_diff'] <= 0.0523),'oe_all_ptile'] = 65
all_merge.loc[(all_merge['pred_all_diff'] > 0.0523) & (all_merge['pred_all_diff'] <= 0.0730),'oe_all_ptile'] = 70
all_merge.loc[(all_merge['pred_all_diff'] > 0.0730) & (all_merge['pred_all_diff'] <= 0.1007),'oe_all_ptile'] = 75
all_merge.loc[(all_merge['pred_all_diff'] > 0.1007) & (all_merge['pred_all_diff'] <= 0.1319),'oe_all_ptile'] = 80
all_merge.loc[(all_merge['pred_all_diff'] > 0.1319) & (all_merge['pred_all_diff'] <= 0.1745),'oe_all_ptile'] = 85
all_merge.loc[(all_merge['pred_all_diff'] > 0.1745) & (all_merge['pred_all_diff'] <= 0.2385),'oe_all_ptile'] = 90
all_merge.loc[(all_merge['pred_all_diff'] > 0.2385) & (all_merge['pred_all_diff'] <= 0.3127),'oe_all_ptile'] = 95
all_merge.loc[all_merge['pred_all_diff'] > 0.3127] = 100

all_merge['oe_all_ptile'] = np.where(all_merge['pred_all_small_flag'] == 1, (50 + 0.5*(all_merge['oe_all_ptile'] - 50)), all_merge['oe_all_ptile'])

all_merge['oe_all_estimate'] = 0
all_merge.loc[all_merge['oe_all_ptile'] <= 5,'oe_all_estimate'] = 1.1239
all_merge.loc[(all_merge['oe_all_ptile'] > 5) & (all_merge['oe_all_ptile'] <= 10),'oe_all_estimate'] = 1.1185
all_merge.loc[(all_merge['oe_all_ptile'] > 10) & (all_merge['oe_all_ptile'] <= 15),'oe_all_estimate'] = 1.0937
all_merge.loc[(all_merge['oe_all_ptile'] > 15) & (all_merge['oe_all_ptile'] <= 35),'oe_all_estimate'] = 1.0285
all_merge.loc[(all_merge['oe_all_ptile'] > 35) & (all_merge['oe_all_ptile'] <= 60),'oe_all_estimate'] = 0.9997
all_merge.loc[(all_merge['oe_all_ptile'] > 60) & (all_merge['oe_all_ptile'] <= 85),'oe_all_estimate'] = 0.9706
all_merge.loc[(all_merge['oe_all_ptile'] > 85) & (all_merge['oe_all_ptile'] <= 90),'oe_all_estimate'] = 0.9082
all_merge.loc[(all_merge['oe_all_ptile'] > 90) & (all_merge['oe_all_ptile'] <= 95),'oe_all_estimate'] = 0.8309
all_merge.loc[all_merge['oe_all_ptile'] > 95,'oe_all_estimate'] = 0.7234

betas_med_worst= np.array([0.0579, 0.5239 , -0.5639 , 0.2097 , 0.9480 , -0.0619])

df_med_worst = all_merge[['how_often_home_health_patients_had_to_be_admitted_to_the_hospital', 'def_prof_personel',\
                          'staff_nutrit_fulltime_binary','aide_over_tot_licensed', 'qm_correct_meds_low_binary',\
                          'percent_of_patients_who_reported_that_their_home_health_team_discussed_medicines_pain_and_home_safety_with_them']]

all_merge['logit_med_worst'] = np.dot(betas_med_worst,df_med_worst.T) + 2.8018

all_merge['pred_med_worst'] = 1/(1+np.exp(-all_merge['logit_med_worst']))

betas_med_best= np.array([-0.0955,-0.0589,0.5094,0.4842,3.1146,0.0451])

df_med_best = all_merge[['how_often_home_health_patients_had_to_be_admitted_to_the_hospital', 'def_total',\
                          'qm_immun_shots_high_binary', 'qm_improv_wounds_high_binary', 'qm_urgent_care_low_binary',\
                          'percent_of_patients_who_reported_that_their_home_health_team_discussed_medicines_pain_and_home_safety_with_them']]

all_merge['logit_med_best'] = np.dot(betas_med_best,df_med_best.T) + -3.8765

all_merge['pred_med_best'] = 1/(1+np.exp(-all_merge['logit_med_best']))

all_merge['pred_med_diff'] = all_merge['pred_med_best'] - all_merge['pred_med_worst']

all_merge['oe_med_ptile'] = 0
all_merge.loc[all_merge['pred_med_diff'] <= -0.2704,'oe_med_ptile'] = 5
all_merge.loc[(all_merge['pred_med_diff'] > -0.2704) & (all_merge['pred_med_diff'] <= -0.2052),'oe_med_ptile'] = 10
all_merge.loc[(all_merge['pred_med_diff'] > -0.2052) & (all_merge['pred_med_diff'] <= -0.1577),'oe_med_ptile'] = 15
all_merge.loc[(all_merge['pred_med_diff'] > -0.1577) & (all_merge['pred_med_diff'] <= -0.1299),'oe_med_ptile'] = 20
all_merge.loc[(all_merge['pred_med_diff'] > -0.1299) & (all_merge['pred_med_diff'] <= -0.0917),'oe_med_ptile'] = 25
all_merge.loc[(all_merge['pred_med_diff'] > -0.0917) & (all_merge['pred_med_diff'] <= -0.0725),'oe_med_ptile'] = 30
all_merge.loc[(all_merge['pred_med_diff'] > -0.0725) & (all_merge['pred_med_diff'] <= -0.0537),'oe_med_ptile'] = 35
all_merge.loc[(all_merge['pred_med_diff'] > -0.0537) & (all_merge['pred_med_diff'] <= -0.0347),'oe_med_ptile'] = 40
all_merge.loc[(all_merge['pred_med_diff'] > -0.0347) & (all_merge['pred_med_diff'] <= -0.0156),'oe_med_ptile'] = 45
all_merge.loc[(all_merge['pred_med_diff'] > -0.0156) & (all_merge['pred_med_diff'] <= 0.0018),'oe_med_ptile'] = 50
all_merge.loc[(all_merge['pred_med_diff'] > 0.0018) & (all_merge['pred_med_diff'] <= 0.0184),'oe_med_ptile'] = 55
all_merge.loc[(all_merge['pred_med_diff'] > 0.0184) & (all_merge['pred_med_diff'] <= 0.0366),'oe_med_ptile'] = 60
all_merge.loc[(all_merge['pred_med_diff'] > 0.0366) & (all_merge['pred_med_diff'] <= 0.0567),'oe_med_ptile'] = 65
all_merge.loc[(all_merge['pred_med_diff'] > 0.0567) & (all_merge['pred_med_diff'] <= 0.0839),'oe_med_ptile'] = 70
all_merge.loc[(all_merge['pred_med_diff'] > 0.0839) & (all_merge['pred_med_diff'] <= 0.1067),'oe_med_ptile'] = 75
all_merge.loc[(all_merge['pred_med_diff'] > 0.1067) & (all_merge['pred_med_diff'] <= 0.1355),'oe_med_ptile'] = 80
all_merge.loc[(all_merge['pred_med_diff'] > 0.1355) & (all_merge['pred_med_diff'] <= 0.1837),'oe_med_ptile'] = 85
all_merge.loc[(all_merge['pred_med_diff'] > 0.1837) & (all_merge['pred_med_diff'] <= 0.2317),'oe_med_ptile'] = 90
all_merge.loc[(all_merge['pred_med_diff'] > 0.2317) & (all_merge['pred_med_diff'] <= 0.3278),'oe_med_ptile'] = 95
all_merge.loc[all_merge['pred_med_diff'] > 0.3278,'oe_med_ptile'] = 100

all_merge['oe_med_ptile'] = np.where(all_merge['pred_med_other_small_flag'] == 1, (50 + 0.5*(all_merge['oe_med_ptile'] - 50)), all_merge['oe_med_ptile'])

all_merge['oe_med_estimate'] = 0
all_merge.loc[all_merge['oe_med_ptile'] <= 5, 'oe_med_estimate'] = 1.1216
all_merge.loc[(all_merge['oe_med_ptile'] > 5) & (all_merge['oe_med_ptile'] <= 10), 'oe_med_estimate'] = 1.0965
all_merge.loc[(all_merge['oe_med_ptile'] > 10) & (all_merge['oe_med_ptile'] <= 15), 'oe_med_estimate'] = 1.0511
all_merge.loc[(all_merge['oe_med_ptile'] > 15) & (all_merge['oe_med_ptile'] <= 50), 'oe_med_estimate'] = 1.0319
all_merge.loc[(all_merge['oe_med_ptile'] > 50) & (all_merge['oe_med_ptile'] <= 75), 'oe_med_estimate'] = 0.9677
all_merge.loc[(all_merge['oe_med_ptile'] > 75) & (all_merge['oe_med_ptile'] <= 85), 'oe_med_estimate'] = 0.9547
all_merge.loc[(all_merge['oe_med_ptile'] > 85) & (all_merge['oe_med_ptile'] <= 90), 'oe_med_estimate'] = 0.8849
all_merge.loc[(all_merge['oe_med_ptile'] > 90) & (all_merge['oe_med_ptile'] <= 95), 'oe_med_estimate'] = 0.8584
all_merge.loc[all_merge['oe_med_ptile'] > 95, 'oe_med_estimate'] = 0.7188



betas_surg_worst = np.array([0.0669, 2.5702 , 0.9438 , -0.0562])

df_surg_worst = all_merge[['how_often_home_health_patients_had_to_be_admitted_to_the_hospital', 'no_hha_serv_binary',\
                          'qm_correct_meds_med_binary','percent_of_patients_who_reported_yes_they_would_definitely_recommend_the_home_health_agency_to_friends_and_family']]

all_merge['logit_surg_worst'] = np.dot(betas_surg_worst,df_surg_worst.T) + 2.0216

all_merge['pred_surg_worst'] = 1/(1+np.exp(-all_merge['logit_surg_worst']))

betas_surg_best= np.array([0.0325,0.5786,0.4340,0.4917,0.6795])

df_surg_best = all_merge[['how_often_the_home_health_team_began_their_patients’_care_in_a_timely_manner', 'survey_good_care_binary',\
                          'qm_immun_shots_high_binary', 'qm_improv_wounds_high_binary', 'qm_urgent_care_high_binary']]

all_merge['logit_surg_best'] = np.dot(betas_surg_best,df_surg_best.T) + -4.9862

all_merge['pred_surg_best'] = 1/(1+np.exp(-all_merge['logit_surg_best']))

all_merge['pred_surg_diff'] = all_merge['pred_surg_best'] - all_merge['pred_surg_worst']

all_merge['oe_surg_ptile'] = 0
all_merge.loc[all_merge['pred_surg_diff'] <= -0.2145,'oe_surg_ptile'] = 5
all_merge.loc[(all_merge['pred_surg_diff'] > -0.2145) & (all_merge['pred_surg_diff'] <= -0.1581),'oe_surg_ptile'] = 10
all_merge.loc[(all_merge['pred_surg_diff'] > -0.1581) & (all_merge['pred_surg_diff'] <= -0.1080),'oe_surg_ptile'] = 15
all_merge.loc[(all_merge['pred_surg_diff'] > -0.1080) & (all_merge['pred_surg_diff'] <= -0.0861),'oe_surg_ptile'] = 20
all_merge.loc[(all_merge['pred_surg_diff'] > -0.0861) & (all_merge['pred_surg_diff'] <= -0.0702),'oe_surg_ptile'] = 25
all_merge.loc[(all_merge['pred_surg_diff'] > -0.0702) & (all_merge['pred_surg_diff'] <= -0.0521),'oe_surg_ptile'] = 30
all_merge.loc[(all_merge['pred_surg_diff'] > -0.0521) & (all_merge['pred_surg_diff'] <= -0.0386),'oe_surg_ptile'] = 35
all_merge.loc[(all_merge['pred_surg_diff'] > -0.0386) & (all_merge['pred_surg_diff'] <= -0.0245),'oe_surg_ptile'] = 40
all_merge.loc[(all_merge['pred_surg_diff'] > -0.0245) & (all_merge['pred_surg_diff'] <= -0.0143),'oe_surg_ptile'] = 45
all_merge.loc[(all_merge['pred_surg_diff'] > -0.0143) & (all_merge['pred_surg_diff'] <= -0.0021),'oe_surg_ptile'] = 50
all_merge.loc[(all_merge['pred_surg_diff'] > -0.0021) & (all_merge['pred_surg_diff'] <= 0.0107),'oe_surg_ptile'] = 55
all_merge.loc[(all_merge['pred_surg_diff'] > 0.0107) & (all_merge['pred_surg_diff'] <= 0.0293),'oe_surg_ptile'] = 60
all_merge.loc[(all_merge['pred_surg_diff'] > 0.0293) & (all_merge['pred_surg_diff'] <= 0.0456),'oe_surg_ptile'] = 65
all_merge.loc[(all_merge['pred_surg_diff'] > 0.0456) & (all_merge['pred_surg_diff'] <= 0.0632),'oe_surg_ptile'] = 70
all_merge.loc[(all_merge['pred_surg_diff'] > 0.0632) & (all_merge['pred_surg_diff'] <= 0.0824),'oe_surg_ptile'] = 75
all_merge.loc[(all_merge['pred_surg_diff'] > 0.0824) & (all_merge['pred_surg_diff'] <= 0.1127),'oe_surg_ptile'] = 80
all_merge.loc[(all_merge['pred_surg_diff'] > 0.1127) & (all_merge['pred_surg_diff'] <= 0.1314),'oe_surg_ptile'] = 85
all_merge.loc[(all_merge['pred_surg_diff'] > 0.1314) & (all_merge['pred_surg_diff'] <= 0.1604),'oe_surg_ptile'] = 90
all_merge.loc[(all_merge['pred_surg_diff'] > 0.1604) & (all_merge['pred_surg_diff'] <= 0.2145),'oe_surg_ptile'] = 95
all_merge.loc[all_merge['pred_surg_diff'] > 0.2145,'oe_surg_ptile'] = 100

all_merge['oe_surg_ptile'] = np.where(all_merge['pred_surg_other_small_flag'] == 1, (50 + 0.5*(all_merge['oe_surg_ptile'] - 50)), all_merge['oe_surg_ptile'])

all_merge['oe_surg_estimate'] = 0
all_merge.loc[all_merge['oe_surg_ptile'] <= 5, 'oe_surg_estimate'] = 1.2346
all_merge.loc[(all_merge['oe_surg_ptile'] > 5) & (all_merge['oe_surg_ptile'] <= 15), 'oe_surg_estimate'] = 1.1099
all_merge.loc[(all_merge['oe_surg_ptile'] > 15) & (all_merge['oe_surg_ptile'] <= 20), 'oe_surg_estimate'] = 1.0235
all_merge.loc[(all_merge['oe_surg_ptile'] > 20) & (all_merge['oe_surg_ptile'] <= 35), 'oe_surg_estimate'] = 1.0142
all_merge.loc[(all_merge['oe_surg_ptile'] > 35) & (all_merge['oe_surg_ptile'] <= 40), 'oe_surg_estimate'] = 1.0060
all_merge.loc[(all_merge['oe_surg_ptile'] > 40) & (all_merge['oe_surg_ptile'] <= 90), 'oe_surg_estimate'] = 0.9772
all_merge.loc[(all_merge['oe_surg_ptile'] > 90) & (all_merge['oe_surg_ptile'] <= 95), 'oe_surg_estimate'] = 0.8264
all_merge.loc[all_merge['oe_surg_ptile'] > 95, 'oe_surg_estimate'] = 0.7276




betas_chf_worst = np.array([0.0579, 11.0786])

df_chf_worst = all_merge[['how_often_home_health_patients_had_to_be_admitted_to_the_hospital', 'rate_adj']]

all_merge['logit_chf_worst'] = np.dot(betas_chf_worst,df_chf_worst.T) + -4.3468

all_merge['pred_chf_worst'] = 1/(1+np.exp(-all_merge['logit_chf_worst']))

betas_chf_best= np.array([-0.0530,-0.9435,-0.6635])

df_chf_best = all_merge[['how_often_home_health_patients_had_to_be_admitted_to_the_hospital', 'alleg_binary',\
                          'def_med_poc_bin']]

all_merge['logit_chf_best'] = np.dot(betas_chf_best,df_chf_best.T) + -0.0274

all_merge['pred_chf_best'] = 1/(1+np.exp(-all_merge['logit_chf_best']))

all_merge['pred_chf_diff'] = all_merge['pred_chf_best'] - all_merge['pred_chf_worst']

all_merge['oe_chf_ptile'] = 0
all_merge.loc[all_merge['pred_chf_diff'] <= -0.2212,'oe_chf_ptile'] = 5
all_merge.loc[(all_merge['pred_chf_diff'] > -0.2212) & (all_merge['pred_chf_diff'] <= -0.1571),'oe_chf_ptile'] = 10
all_merge.loc[(all_merge['pred_chf_diff'] > -0.1571) & (all_merge['pred_chf_diff'] <= -0.1268),'oe_chf_ptile'] = 15
all_merge.loc[(all_merge['pred_chf_diff'] > -0.1268) & (all_merge['pred_chf_diff'] <= -0.1039),'oe_chf_ptile'] = 20
all_merge.loc[(all_merge['pred_chf_diff'] > -0.1039) & (all_merge['pred_chf_diff'] <= -0.0845),'oe_chf_ptile'] = 25
all_merge.loc[(all_merge['pred_chf_diff'] > -0.0845) & (all_merge['pred_chf_diff'] <= -0.0695),'oe_chf_ptile'] = 30
all_merge.loc[(all_merge['pred_chf_diff'] > -0.0695) & (all_merge['pred_chf_diff'] <= -0.0652),'oe_chf_ptile'] = 35
all_merge.loc[(all_merge['pred_chf_diff'] > -0.0652) & (all_merge['pred_chf_diff'] <= -0.0461),'oe_chf_ptile'] = 40
all_merge.loc[(all_merge['pred_chf_diff'] > -0.0461) & (all_merge['pred_chf_diff'] <= -0.0273),'oe_chf_ptile'] = 45
all_merge.loc[(all_merge['pred_chf_diff'] > -0.0273) & (all_merge['pred_chf_diff'] <= -0.0085),'oe_chf_ptile'] = 50
all_merge.loc[(all_merge['pred_chf_diff'] > -0.0085) & (all_merge['pred_chf_diff'] <= 0.0100),'oe_chf_ptile'] = 55
all_merge.loc[(all_merge['pred_chf_diff'] > 0.0100) & (all_merge['pred_chf_diff'] <= 0.0283),'oe_chf_ptile'] = 60
all_merge.loc[(all_merge['pred_chf_diff'] > 0.0283) & (all_merge['pred_chf_diff'] <= 0.0456),'oe_chf_ptile'] = 65
all_merge.loc[(all_merge['pred_chf_diff'] > 0.0456) & (all_merge['pred_chf_diff'] <= 0.0646),'oe_chf_ptile'] = 70
all_merge.loc[(all_merge['pred_chf_diff'] > 0.0646) & (all_merge['pred_chf_diff'] <= 0.0824),'oe_chf_ptile'] = 75
all_merge.loc[(all_merge['pred_chf_diff'] > 0.0824) & (all_merge['pred_chf_diff'] <= 0.1002),'oe_chf_ptile'] = 80
all_merge.loc[(all_merge['pred_chf_diff'] > 0.1002) & (all_merge['pred_chf_diff'] <= 0.1339),'oe_chf_ptile'] = 85
all_merge.loc[(all_merge['pred_chf_diff'] > 0.1339) & (all_merge['pred_chf_diff'] <= 0.1768),'oe_chf_ptile'] = 90
all_merge.loc[(all_merge['pred_chf_diff'] > 0.1768) & (all_merge['pred_chf_diff'] <= 0.2210),'oe_chf_ptile'] = 95
all_merge.loc[all_merge['pred_chf_diff'] > 0.2210,'oe_chf_ptile'] = 100

all_merge['oe_chf_ptile'] = np.where(all_merge['pred_CHF_small_flag'] == 1, (50 + 0.5*(all_merge['oe_chf_ptile'] - 50)), all_merge['oe_chf_ptile'])

all_merge['oe_chf_estimate'] = 0
all_merge.loc[all_merge['oe_chf_ptile'] <= 15, 'oe_chf_estimate'] = 1.1194
all_merge.loc[(all_merge['oe_chf_ptile'] > 15) & (all_merge['oe_chf_ptile'] <= 25), 'oe_chf_estimate'] = 1.0585
all_merge.loc[(all_merge['oe_chf_ptile'] > 25) & (all_merge['oe_chf_ptile'] <= 55), 'oe_chf_estimate'] = 0.9903
all_merge.loc[(all_merge['oe_chf_ptile'] > 55) & (all_merge['oe_chf_ptile'] <= 80), 'oe_chf_estimate'] = 0.9596
all_merge.loc[(all_merge['oe_chf_ptile'] > 80) & (all_merge['oe_chf_ptile'] <= 95), 'oe_chf_estimate'] = 0.9482
all_merge.loc[all_merge['oe_chf_ptile'] > 95, 'oe_chf_estimate'] = 0.8265



betas_med_high_worst = np.array([0.0938, 0.2065, 0.000007907, -0.6075, -0.7342, -0.0703])

df_med_high_worst = all_merge[['how_often_home_health_patients_had_to_be_admitted_to_the_hospital', 'def_org_admin',\
                               'census_discharge_hosp_per_year', 'staff_ot_fulltime_binary', 'qm_diab_care_high_binary',
                               'percent_of_patients_who_reported_that_their_home_health_team_discussed_medicines_pain_and_home_safety_with_them']]

all_merge['logit_med_high_worst'] = np.dot(betas_med_high_worst,df_med_high_worst.T) + 3.2519

all_merge['pred_med_high_worst'] = 1/(1+np.exp(-all_merge['logit_med_high_worst']))

betas_med_high_best= np.array([-0.1091,0.8109,0.7478])

df_med_high_best = all_merge[['how_often_home_health_patients_had_to_be_admitted_to_the_hospital', 'qm_diab_care_high_binary',\
                          'qm_improv_wounds_high_binary']]

all_merge['logit_med_high_best'] = np.dot(betas_med_high_best,df_med_high_best.T) + -0.1606

all_merge['pred_med_high_best'] = 1/(1+np.exp(-all_merge['logit_med_high_best']))

all_merge['pred_med_high_diff'] = all_merge['pred_med_high_best'] - all_merge['pred_med_high_worst']

all_merge['oe_med_high_ptile'] = 0
all_merge.loc[all_merge['pred_med_high_diff'] <= -0.3364,'oe_med_high_ptile'] = 5
all_merge.loc[(all_merge['pred_med_high_diff'] > -0.3364) & (all_merge['pred_med_high_diff'] <= -0.2563),'oe_med_high_ptile'] = 10
all_merge.loc[(all_merge['pred_med_high_diff'] > -0.2563) & (all_merge['pred_med_high_diff'] <= -0.1983),'oe_med_high_ptile'] = 15
all_merge.loc[(all_merge['pred_med_high_diff'] > -0.1983) & (all_merge['pred_med_high_diff'] <= -0.1615),'oe_med_high_ptile'] = 20
all_merge.loc[(all_merge['pred_med_high_diff'] > -0.1615) & (all_merge['pred_med_high_diff'] <= -0.1361),'oe_med_high_ptile'] = 25
all_merge.loc[(all_merge['pred_med_high_diff'] > -0.1361) & (all_merge['pred_med_high_diff'] <= -0.1047),'oe_med_high_ptile'] = 30
all_merge.loc[(all_merge['pred_med_high_diff'] > -0.1047) & (all_merge['pred_med_high_diff'] <= -0.0690),'oe_med_high_ptile'] = 35
all_merge.loc[(all_merge['pred_med_high_diff'] > -0.0690) & (all_merge['pred_med_high_diff'] <= -0.0414),'oe_med_high_ptile'] = 40
all_merge.loc[(all_merge['pred_med_high_diff'] > -0.0414) & (all_merge['pred_med_high_diff'] <= -0.0199),'oe_med_high_ptile'] = 45
all_merge.loc[(all_merge['pred_med_high_diff'] > -0.0199) & (all_merge['pred_med_high_diff'] <= -0.0019),'oe_med_high_ptile'] = 50
all_merge.loc[(all_merge['pred_med_high_diff'] > -0.0019) & (all_merge['pred_med_high_diff'] <= 0.0156),'oe_med_high_ptile'] = 55
all_merge.loc[(all_merge['pred_med_high_diff'] > 0.0156) & (all_merge['pred_med_high_diff'] <= 0.0377),'oe_med_high_ptile'] = 60
all_merge.loc[(all_merge['pred_med_high_diff'] > 0.0377) & (all_merge['pred_med_high_diff'] <= 0.0634),'oe_med_high_ptile'] = 65
all_merge.loc[(all_merge['pred_med_high_diff'] > 0.0634) & (all_merge['pred_med_high_diff'] <= 0.1008),'oe_med_high_ptile'] = 70
all_merge.loc[(all_merge['pred_med_high_diff'] > 0.1008) & (all_merge['pred_med_high_diff'] <= 0.1361),'oe_med_high_ptile'] = 75
all_merge.loc[(all_merge['pred_med_high_diff'] > 0.1361) & (all_merge['pred_med_high_diff'] <= 0.1710),'oe_med_high_ptile'] = 80
all_merge.loc[(all_merge['pred_med_high_diff'] > 0.1710) & (all_merge['pred_med_high_diff'] <= 0.2105),'oe_med_high_ptile'] = 85
all_merge.loc[(all_merge['pred_med_high_diff'] > 0.2105) & (all_merge['pred_med_high_diff'] <= 0.2758),'oe_med_high_ptile'] = 90
all_merge.loc[(all_merge['pred_med_high_diff'] > 0.2758) & (all_merge['pred_med_high_diff'] <= 0.3952),'oe_med_high_ptile'] = 95
all_merge.loc[all_merge['pred_med_high_diff'] > 0.3952,'oe_med_high_ptile'] = 100

all_merge['oe_med_high_ptile'] = np.where(all_merge['pred_med_high_risk_small_flag'] == 1, (50 + 0.5*(all_merge['oe_med_high_ptile'] - 50)), all_merge['oe_med_high_ptile'])

all_merge['oe_med_high_estimate'] = 0
all_merge.loc[all_merge['oe_med_high_ptile'] <= 5, 'oe_med_high_estimate'] = 1.1396
all_merge.loc[(all_merge['oe_med_high_ptile'] > 5) & (all_merge['oe_med_high_ptile'] <= 15), 'oe_med_high_estimate'] = 1.0689
all_merge.loc[(all_merge['oe_med_high_ptile'] > 15) & (all_merge['oe_med_high_ptile'] <= 70), 'oe_med_high_estimate'] = 1.0052
all_merge.loc[(all_merge['oe_med_high_ptile'] > 70) & (all_merge['oe_med_high_ptile'] <= 85), 'oe_med_high_estimate'] = 0.9152
all_merge.loc[(all_merge['oe_med_high_ptile'] > 85) & (all_merge['oe_med_high_ptile'] <= 90), 'oe_med_high_estimate'] = 0.8747
all_merge.loc[(all_merge['oe_med_high_ptile'] > 90) & (all_merge['oe_med_high_ptile'] <= 95), 'oe_med_high_estimate'] = 0.8254
all_merge.loc[all_merge['oe_med_high_ptile'] > 95, 'oe_med_high_estimate'] = 0.7884



betas_med_low_worst = np.array([-0.0256, 0.0366, 0.6124, 0.5877, 0.6276])

df_med_low_worst = all_merge[['how_often_patients_got_better_at_taking_their_drugs_correctly_by_mouth',\
                             'how_often_home_health_patients_had_to_be_admitted_to_the_hospital',\
                          'for_profit_binary','medicare_only_binary','qm_diab_care_low_binary']]

all_merge['logit_med_low_worst'] = np.dot(betas_med_low_worst,df_med_low_worst.T) + -1.2702

all_merge['pred_med_low_worst'] = 1/(1+np.exp(-all_merge['logit_med_low_worst']))

betas_med_low_best= np.array([0.0372,-0.0702,0.3554,0.0488])

df_med_low_best = all_merge[['how_often_patients_got_better_at_taking_their_drugs_correctly_by_mouth',\
                             'how_often_home_health_patients_had_to_be_admitted_to_the_hospital',\
                          'survey_good_care_binary','percent_of_patients_who_reported_that_their_home_health_team_discussed_medicines_pain_and_home_safety_with_them']]

all_merge['logit_med_low_best'] = np.dot(betas_med_low_best,df_med_low_best.T) + -6.6563

all_merge['pred_med_low_best'] = 1/(1+np.exp(-all_merge['logit_med_low_best']))

all_merge['pred_med_low_diff'] = all_merge['pred_med_low_best'] - all_merge['pred_med_low_worst']

all_merge['oe_med_low_ptile'] = 0
all_merge.loc[all_merge['pred_med_low_diff'] <= 0.2521,'oe_med_low_ptile'] = 5
all_merge.loc[(all_merge['pred_med_low_diff'] > 0.2521) & (all_merge['pred_med_low_diff'] <= -0.1971),'oe_med_low_ptile'] = 10
all_merge.loc[(all_merge['pred_med_low_diff'] > -0.1971) & (all_merge['pred_med_low_diff'] <= -0.1484),'oe_med_low_ptile'] = 15
all_merge.loc[(all_merge['pred_med_low_diff'] > -0.1484) & (all_merge['pred_med_low_diff'] <= -0.1191),'oe_med_low_ptile'] = 20
all_merge.loc[(all_merge['pred_med_low_diff'] > -0.1191) & (all_merge['pred_med_low_diff'] <= -0.0961),'oe_med_low_ptile'] = 25
all_merge.loc[(all_merge['pred_med_low_diff'] > -0.0961) & (all_merge['pred_med_low_diff'] <= -0.0799),'oe_med_low_ptile'] = 30
all_merge.loc[(all_merge['pred_med_low_diff'] > -0.0799) & (all_merge['pred_med_low_diff'] <= -0.0568),'oe_med_low_ptile'] = 35
all_merge.loc[(all_merge['pred_med_low_diff'] > -0.0568) & (all_merge['pred_med_low_diff'] <= -0.0387),'oe_med_low_ptile'] = 40
all_merge.loc[(all_merge['pred_med_low_diff'] > -0.0387) & (all_merge['pred_med_low_diff'] <= -0.0174),'oe_med_low_ptile'] = 45
all_merge.loc[(all_merge['pred_med_low_diff'] > -0.0174) & (all_merge['pred_med_low_diff'] <= 0.0026),'oe_med_low_ptile'] = 50
all_merge.loc[(all_merge['pred_med_low_diff'] > 0.0026) & (all_merge['pred_med_low_diff'] <= 0.0177),'oe_med_low_ptile'] = 55
all_merge.loc[(all_merge['pred_med_low_diff'] > 0.0177) & (all_merge['pred_med_low_diff'] <= 0.0385),'oe_med_low_ptile'] = 60
all_merge.loc[(all_merge['pred_med_low_diff'] > 0.0385) & (all_merge['pred_med_low_diff'] <= 0.0596),'oe_med_low_ptile'] = 65
all_merge.loc[(all_merge['pred_med_low_diff'] > 0.0596) & (all_merge['pred_med_low_diff'] <= 0.0733),'oe_med_low_ptile'] = 70
all_merge.loc[(all_merge['pred_med_low_diff'] > 0.0733) & (all_merge['pred_med_low_diff'] <= 0.0948),'oe_med_low_ptile'] = 75
all_merge.loc[(all_merge['pred_med_low_diff'] > 0.0948) & (all_merge['pred_med_low_diff'] <= 0.1258),'oe_med_low_ptile'] = 80
all_merge.loc[(all_merge['pred_med_low_diff'] > 0.1258) & (all_merge['pred_med_low_diff'] <= 0.1635),'oe_med_low_ptile'] = 85
all_merge.loc[(all_merge['pred_med_low_diff'] > 0.1635) & (all_merge['pred_med_low_diff'] <= 0.2066),'oe_med_low_ptile'] = 90
all_merge.loc[(all_merge['pred_med_low_diff'] > 0.2066) & (all_merge['pred_med_low_diff'] <= 0.2577),'oe_med_low_ptile'] = 95
all_merge.loc[all_merge['pred_med_low_diff'] > 0.2577,'oe_med_low_ptile'] = 100

all_merge['oe_med_low_ptile'] = np.where(all_merge['pred_med_low_risk_small_flag'] == 1, (50 + 0.5*(all_merge['oe_med_low_ptile'] - 50)), all_merge['oe_med_low_ptile'])

all_merge['oe_med_low_estimate']= 0
all_merge.loc[all_merge['oe_med_low_ptile'] <= 30, 'oe_med_low_estimate'] = 1.0701
all_merge.loc[(all_merge['oe_med_low_ptile'] > 30) & (all_merge['oe_med_low_ptile'] <= 45), 'oe_med_low_estimate'] = 1.0456
all_merge.loc[(all_merge['oe_med_low_ptile'] > 45) & (all_merge['oe_med_low_ptile'] <= 80), 'oe_med_low_estimate'] = 0.9609
all_merge.loc[(all_merge['oe_med_low_ptile'] > 80) & (all_merge['oe_med_low_ptile'] <= 85), 'oe_med_low_estimate'] = 0.8865
all_merge.loc[(all_merge['oe_med_low_ptile'] > 85) & (all_merge['oe_med_low_ptile'] <= 90), 'oe_med_low_estimate'] = 0.8746
all_merge.loc[(all_merge['oe_med_low_ptile'] > 90) & (all_merge['oe_med_low_ptile'] <= 95), 'oe_med_low_estimate'] = 0.8658
all_merge.loc[all_merge['oe_med_low_ptile'] > 95, 'oe_med_low_estimate'] = 0.7931



betas_surg_high_worst = np.array([0.0759, -0.9539, 3.2871, -0.0596])

df_surg_high_worst = all_merge[['how_often_home_health_patients_had_to_be_admitted_to_the_hospital',
                             'staff_nutrit_fulltime_binary',
                          'no_hha_serv_binary','percent_of_patients_who_reported_yes_they_would_definitely_recommend_the_home_health_agency_to_friends_and_family']]

all_merge['logit_surg_high_worst'] = np.dot(betas_surg_high_worst,df_surg_high_worst.T) + 2.3094

all_merge['pred_surg_high_worst'] = 1/(1+np.exp(-all_merge['logit_surg_high_worst']))

betas_surg_high_best= np.array([0.4770,0.5348,0.5991])

df_surg_high_best = all_merge[['survey_good_care_binary',
                             'qm_diab_care_high_binary',
                          'qm_improv_wounds_high_binary']]

all_merge['logit_surg_high_best'] = np.dot(betas_surg_high_best,df_surg_high_best.T) + -1.8662

all_merge['pred_surg_high_best'] = 1/(1+np.exp(-all_merge['logit_surg_high_best']))

all_merge['pred_surg_high_diff'] = all_merge['pred_surg_high_best'] - all_merge['pred_surg_high_worst']

all_merge['oe_surg_high_ptile'] = 0
all_merge.loc[all_merge['pred_surg_high_diff'] <= -0.2430,'oe_surg_high_ptile'] = 5
all_merge.loc[(all_merge['pred_surg_high_diff'] > -0.2430) & (all_merge['pred_surg_high_diff'] <= -0.1488),'oe_surg_high_ptile'] = 10
all_merge.loc[(all_merge['pred_surg_high_diff'] > -0.1488) & (all_merge['pred_surg_high_diff'] <= -0.1081),'oe_surg_high_ptile'] = 15
all_merge.loc[(all_merge['pred_surg_high_diff'] > -0.1081) & (all_merge['pred_surg_high_diff'] <= -0.0867),'oe_surg_high_ptile'] = 20
all_merge.loc[(all_merge['pred_surg_high_diff'] > -0.0867) & (all_merge['pred_surg_high_diff'] <= -0.0694),'oe_surg_high_ptile'] = 25
all_merge.loc[(all_merge['pred_surg_high_diff'] > -0.0694) & (all_merge['pred_surg_high_diff'] <= -0.0469),'oe_surg_high_ptile'] = 30
all_merge.loc[(all_merge['pred_surg_high_diff'] > -0.0469) & (all_merge['pred_surg_high_diff'] <= -0.0300),'oe_surg_high_ptile'] = 35
all_merge.loc[(all_merge['pred_surg_high_diff'] > -0.0300) & (all_merge['pred_surg_high_diff'] <= -0.0157),'oe_surg_high_ptile'] = 40
all_merge.loc[(all_merge['pred_surg_high_diff'] > -0.0157) & (all_merge['pred_surg_high_diff'] <= 0.0026),'oe_surg_high_ptile'] = 45
all_merge.loc[(all_merge['pred_surg_high_diff'] > 0.0026) & (all_merge['pred_surg_high_diff'] <= 0.0162),'oe_surg_high_ptile'] = 50
all_merge.loc[(all_merge['pred_surg_high_diff'] > 0.0162) & (all_merge['pred_surg_high_diff'] <= 0.0295),'oe_surg_high_ptile'] = 55
all_merge.loc[(all_merge['pred_surg_high_diff'] > 0.0295) & (all_merge['pred_surg_high_diff'] <= 0.0443),'oe_surg_high_ptile'] = 60
all_merge.loc[(all_merge['pred_surg_high_diff'] > 0.0443) & (all_merge['pred_surg_high_diff'] <= 0.0593),'oe_surg_high_ptile'] = 65
all_merge.loc[(all_merge['pred_surg_high_diff'] > 0.0593) & (all_merge['pred_surg_high_diff'] <= 0.0743),'oe_surg_high_ptile'] = 70
all_merge.loc[(all_merge['pred_surg_high_diff'] > 0.0743) & (all_merge['pred_surg_high_diff'] <= 0.0910),'oe_surg_high_ptile'] = 75
all_merge.loc[(all_merge['pred_surg_high_diff'] > 0.0910) & (all_merge['pred_surg_high_diff'] <= 0.1085),'oe_surg_high_ptile'] = 80
all_merge.loc[(all_merge['pred_surg_high_diff'] > 0.1085) & (all_merge['pred_surg_high_diff'] <= 0.1275),'oe_surg_high_ptile'] = 85
all_merge.loc[(all_merge['pred_surg_high_diff'] > 0.1275) & (all_merge['pred_surg_high_diff'] <= 0.1453),'oe_surg_high_ptile'] = 90
all_merge.loc[(all_merge['pred_surg_high_diff'] > 0.1453) & (all_merge['pred_surg_high_diff'] <= 0.1921),'oe_surg_high_ptile'] = 95
all_merge.loc[all_merge['pred_surg_high_diff'] > 0.1921,'oe_surg_high_ptile'] = 100

all_merge['oe_surg_high_ptile'] = np.where(all_merge['pred_surg_high_risk_small_flag'] == 1, (50 + 0.5*(all_merge['oe_surg_high_ptile'] - 50)), all_merge['oe_surg_high_ptile'])

all_merge['oe_surg_high_estimate']= 0
all_merge.loc[all_merge['oe_surg_high_ptile'] <= 5, 'oe_surg_high_estimate'] = 1.2470
all_merge.loc[(all_merge['oe_surg_high_ptile'] > 5) & (all_merge['oe_surg_high_ptile'] <= 10), 'oe_surg_high_estimate'] = 1.1841
all_merge.loc[(all_merge['oe_surg_high_ptile'] > 10) & (all_merge['oe_surg_high_ptile'] <= 70), 'oe_surg_high_estimate'] = 1.0464
all_merge.loc[(all_merge['oe_surg_high_ptile'] > 70) & (all_merge['oe_surg_high_ptile'] <= 95), 'oe_surg_high_estimate'] = 0.9019
all_merge.loc[all_merge['oe_surg_high_ptile'] > 95, 'oe_surg_high_estimate'] = 0.7995






betas_surg_low_worst = np.array([0.7661, 0.9542, 0.7271, -0.5405,-0.7840,3.0103,-0.0314,2.6364])

df_surg_low_worst = all_merge[['def_prof_personel',
                             'def_report_info',
                          'for_profit_binary','staff_speech_fulltime_binary',
                          'staff_nutrit_fulltime_binary','no_hha_serv_binary',
                          'qm_immun_shots','qm_func_status_avg_med_binary']]

all_merge['logit_surg_low_worst'] = np.dot(betas_surg_low_worst,df_surg_low_worst.T) + 0.4902

all_merge['pred_surg_low_worst'] = 1/(1+np.exp(-all_merge['logit_surg_low_worst']))

betas_surg_low_best= np.array([-0.0646])

df_surg_low_best = all_merge[['how_often_home_health_patients_had_to_be_admitted_to_the_hospital']]

all_merge['logit_surg_low_best'] = np.dot(betas_surg_low_best,df_surg_low_best.T) + -0.4813

all_merge['pred_surg_low_best'] = 1/(1+np.exp(-all_merge['logit_surg_low_best']))

all_merge['pred_surg_low_diff'] = all_merge['pred_surg_low_best'] - all_merge['pred_surg_low_worst']

all_merge['oe_surg_low_ptile'] = 0
all_merge.loc[all_merge['pred_surg_low_diff'] <= -0.2896,'oe_surg_low_ptile'] = 5
all_merge.loc[(all_merge['pred_surg_low_diff'] > -0.2896) & (all_merge['pred_surg_low_diff'] <= -0.1776),'oe_surg_low_ptile'] = 10
all_merge.loc[(all_merge['pred_surg_low_diff'] > -0.1776) & (all_merge['pred_surg_low_diff'] <= -0.1355),'oe_surg_low_ptile'] = 15
all_merge.loc[(all_merge['pred_surg_low_diff'] > -0.1355) & (all_merge['pred_surg_low_diff'] <= -0.1052),'oe_surg_low_ptile'] = 20
all_merge.loc[(all_merge['pred_surg_low_diff'] > -0.1052) & (all_merge['pred_surg_low_diff'] <= -0.0775),'oe_surg_low_ptile'] = 25
all_merge.loc[(all_merge['pred_surg_low_diff'] > -0.0775) & (all_merge['pred_surg_low_diff'] <= -0.0580),'oe_surg_low_ptile'] = 30
all_merge.loc[(all_merge['pred_surg_low_diff'] > -0.0580) & (all_merge['pred_surg_low_diff'] <= -0.0430),'oe_surg_low_ptile'] = 35
all_merge.loc[(all_merge['pred_surg_low_diff'] > -0.0430) & (all_merge['pred_surg_low_diff'] <= -0.0240),'oe_surg_low_ptile'] = 40
all_merge.loc[(all_merge['pred_surg_low_diff'] > -0.0240) & (all_merge['pred_surg_low_diff'] <= -0.0041),'oe_surg_low_ptile'] = 45
all_merge.loc[(all_merge['pred_surg_low_diff'] > -0.0041) & (all_merge['pred_surg_low_diff'] <= 0.0181),'oe_surg_low_ptile'] = 50
all_merge.loc[(all_merge['pred_surg_low_diff'] > 0.0181) & (all_merge['pred_surg_low_diff'] <= 0.0366),'oe_surg_low_ptile'] = 55
all_merge.loc[(all_merge['pred_surg_low_diff'] > 0.0366) & (all_merge['pred_surg_low_diff'] <= 0.0578),'oe_surg_low_ptile'] = 60
all_merge.loc[(all_merge['pred_surg_low_diff'] > 0.0578) & (all_merge['pred_surg_low_diff'] <= 0.0717),'oe_surg_low_ptile'] = 65
all_merge.loc[(all_merge['pred_surg_low_diff'] > 0.0717) & (all_merge['pred_surg_low_diff'] <= 0.0871),'oe_surg_low_ptile'] = 70
all_merge.loc[(all_merge['pred_surg_low_diff'] > 0.0871) & (all_merge['pred_surg_low_diff'] <= 0.1046),'oe_surg_low_ptile'] = 75
all_merge.loc[(all_merge['pred_surg_low_diff'] > 0.1046) & (all_merge['pred_surg_low_diff'] <= 0.1309),'oe_surg_low_ptile'] = 80
all_merge.loc[(all_merge['pred_surg_low_diff'] > 0.1309) & (all_merge['pred_surg_low_diff'] <= 0.1575),'oe_surg_low_ptile'] = 85
all_merge.loc[(all_merge['pred_surg_low_diff'] > 0.1575) & (all_merge['pred_surg_low_diff'] <= 0.1837),'oe_surg_low_ptile'] = 90
all_merge.loc[(all_merge['pred_surg_low_diff'] > 0.1837) & (all_merge['pred_surg_low_diff'] <= 0.2146),'oe_surg_low_ptile'] = 95
all_merge.loc[all_merge['pred_surg_low_diff'] > 0.2146,'oe_surg_low_ptile'] = 100

all_merge['oe_surg_low_ptile'] = np.where(all_merge['pred_surg_low_risk_small_flag'] == 1, (50 + 0.5*(all_merge['oe_surg_low_ptile'] - 50)), all_merge['oe_surg_low_ptile'])

all_merge['oe_surg_low_estimate']= 0
all_merge.loc[all_merge['oe_surg_low_ptile'] <= 5, 'oe_surg_low_estimate'] = 1.7016
all_merge.loc[(all_merge['oe_surg_low_ptile'] > 5) & (all_merge['oe_surg_low_ptile'] <= 10), 'oe_surg_low_estimate'] = 1.1543
all_merge.loc[(all_merge['oe_surg_low_ptile'] > 10) & (all_merge['oe_surg_low_ptile'] <= 25), 'oe_surg_low_estimate'] = 1.0696
all_merge.loc[(all_merge['oe_surg_low_ptile'] > 25) & (all_merge['oe_surg_low_ptile'] <= 70), 'oe_surg_low_estimate'] = 1.0034
all_merge.loc[(all_merge['oe_surg_low_ptile'] > 70) & (all_merge['oe_surg_low_ptile'] <= 90), 'oe_surg_low_estimate'] = 0.7357
all_merge.loc[(all_merge['oe_surg_low_ptile'] > 90) & (all_merge['oe_surg_low_ptile'] <= 95), 'oe_surg_low_estimate'] = 0.5936
all_merge.loc[all_merge['oe_surg_low_ptile'] > 95, 'oe_surg_low_estimate'] = 0.5594



betas_lung_worst = np.array([-0.0530, 0.0899, 0.3670])

df_lung_worst = all_merge[['how_often_the_home_health_team_checked_patients_for_depression',
                           'how_often_home_health_patients_had_to_be_admitted_to_the_hospital',
                          'def_org_admin']]

all_merge['logit_lung_worst'] = np.dot(betas_lung_worst,df_lung_worst.T) + 2.1279

all_merge['pred_lung_worst'] = 1/(1+np.exp(-all_merge['logit_lung_worst']))

betas_lung_best= np.array([-0.0978,0.7485,0.7664,1.4365])

df_lung_best = all_merge[['how_often_home_health_patients_had_to_be_admitted_to_the_hospital',
                          'medical_admin_binary', 'qm_diab_care_high_binary', 'qm_func_status_avg_high_binary']]

all_merge['logit_lung_best'] = np.dot(betas_lung_best,df_lung_best.T) + -0.8045

all_merge['pred_lung_best'] = 1/(1+np.exp(-all_merge['logit_lung_best']))

all_merge['pred_lung_diff'] = all_merge['pred_lung_best'] - all_merge['pred_lung_worst']

all_merge['oe_lung_ptile'] = 0
all_merge.loc[all_merge['pred_lung_diff'] <= -0.3230,'oe_lung_ptile'] = 5
all_merge.loc[(all_merge['pred_lung_diff'] > -0.3230) & (all_merge['pred_lung_diff'] <= -0.2329),'oe_lung_ptile'] = 10
all_merge.loc[(all_merge['pred_lung_diff'] > -0.2329) & (all_merge['pred_lung_diff'] <= -0.1827),'oe_lung_ptile'] = 15
all_merge.loc[(all_merge['pred_lung_diff'] > -0.1827) & (all_merge['pred_lung_diff'] <= -0.1334),'oe_lung_ptile'] = 20
all_merge.loc[(all_merge['pred_lung_diff'] > -0.1334) & (all_merge['pred_lung_diff'] <= -0.1031),'oe_lung_ptile'] = 25
all_merge.loc[(all_merge['pred_lung_diff'] > -0.1031) & (all_merge['pred_lung_diff'] <= -0.0840),'oe_lung_ptile'] = 30
all_merge.loc[(all_merge['pred_lung_diff'] > -0.0840) & (all_merge['pred_lung_diff'] <= -0.0593),'oe_lung_ptile'] = 35
all_merge.loc[(all_merge['pred_lung_diff'] > -0.0593) & (all_merge['pred_lung_diff'] <= -0.0368),'oe_lung_ptile'] = 40
all_merge.loc[(all_merge['pred_lung_diff'] > -0.0368) & (all_merge['pred_lung_diff'] <= -0.0172),'oe_lung_ptile'] = 45
all_merge.loc[(all_merge['pred_lung_diff'] > -0.0172) & (all_merge['pred_lung_diff'] <= -0.0048),'oe_lung_ptile'] = 50
all_merge.loc[(all_merge['pred_lung_diff'] > -0.0048) & (all_merge['pred_lung_diff'] <= 0.0183),'oe_lung_ptile'] = 55
all_merge.loc[(all_merge['pred_lung_diff'] > 0.0183) & (all_merge['pred_lung_diff'] <= 0.0400),'oe_lung_ptile'] = 60
all_merge.loc[(all_merge['pred_lung_diff'] > 0.0400) & (all_merge['pred_lung_diff'] <= 0.0601),'oe_lung_ptile'] = 65
all_merge.loc[(all_merge['pred_lung_diff'] > 0.0601) & (all_merge['pred_lung_diff'] <= 0.0851),'oe_lung_ptile'] = 70
all_merge.loc[(all_merge['pred_lung_diff'] > 0.0851) & (all_merge['pred_lung_diff'] <= 0.1057),'oe_lung_ptile'] = 75
all_merge.loc[(all_merge['pred_lung_diff'] > 0.1057) & (all_merge['pred_lung_diff'] <= 0.1433),'oe_lung_ptile'] = 80
all_merge.loc[(all_merge['pred_lung_diff'] > 0.1433) & (all_merge['pred_lung_diff'] <= 0.1774),'oe_lung_ptile'] = 85
all_merge.loc[(all_merge['pred_lung_diff'] > 0.1774) & (all_merge['pred_lung_diff'] <= 0.2265),'oe_lung_ptile'] = 90
all_merge.loc[(all_merge['pred_lung_diff'] > 0.2265) & (all_merge['pred_lung_diff'] <= 0.3355),'oe_lung_ptile'] = 95
all_merge.loc[all_merge['pred_lung_diff'] > 0.3355,'oe_lung_ptile'] = 100

all_merge['oe_lung_ptile'] = np.where(all_merge['pred_lung_small_flag'] == 1, (50 + 0.5*(all_merge['oe_lung_ptile'] - 50)), all_merge['oe_lung_ptile'])

all_merge['oe_lung_estimate']= 0
all_merge.loc[all_merge['oe_lung_ptile'] <= 5, 'oe_lung_estimate'] = 1.1634
all_merge.loc[(all_merge['oe_lung_ptile'] > 5) & (all_merge['oe_lung_ptile'] <= 15), 'oe_lung_estimate'] = 1.0664
all_merge.loc[(all_merge['oe_lung_ptile'] > 15) & (all_merge['oe_lung_ptile'] <= 45), 'oe_lung_estimate'] = 1.0269
all_merge.loc[(all_merge['oe_lung_ptile'] > 45) & (all_merge['oe_lung_ptile'] <= 85), 'oe_lung_estimate'] = 0.9256
all_merge.loc[(all_merge['oe_lung_ptile'] > 85) & (all_merge['oe_lung_ptile'] <= 90), 'oe_lung_estimate'] = 0.8729
all_merge.loc[(all_merge['oe_lung_ptile'] > 90) & (all_merge['oe_lung_ptile'] <= 95), 'oe_lung_estimate'] = 0.8539
all_merge.loc[all_merge['oe_lung_ptile'] > 95, 'oe_lung_estimate'] = 0.7666

all_merge['oe_all_state_median'] = all_merge.groupby('state')['oe_all_estimate'].transform('median')
all_merge['oe_med_state_median'] = all_merge.groupby('state')['oe_med_estimate'].transform('median')
all_merge['oe_med_low_state_median'] = all_merge.groupby('state')['oe_med_low_estimate'].transform('median')
all_merge['oe_med_high_state_median'] = all_merge.groupby('state')['oe_med_high_estimate'].transform('median')
all_merge['oe_chf_state_median'] = all_merge.groupby('state')['oe_chf_estimate'].transform('median')
all_merge['oe_lung_state_median'] = all_merge.groupby('state')['oe_lung_estimate'].transform('median')
all_merge['oe_surg_state_median'] = all_merge.groupby('state')['oe_surg_estimate'].transform('median')
all_merge['oe_surg_low_state_median'] = all_merge.groupby('state')['oe_surg_low_estimate'].transform('median')
all_merge['oe_surg_high_state_median'] = all_merge.groupby('state')['oe_surg_high_estimate'].transform('median')

all_merge['qm_timely_manner_ptile'] = \
(all_merge.groupby('state')['how_often_the_home_health_team_began_their_patients’_care_in_a_timely_manner'].rank(pct=True) * 100).round()

all_merge['qm_teach_drugs_ptile'] = \
(all_merge.groupby('state')['how_often_the_home_health_team_taught_patients_or_their_family_caregivers_about_their_drugs'].rank(pct=True) * 100).round()

all_merge['qm_risk_falling_ptile'] = \
(all_merge.groupby('state')['how_often_the_home_health_team_checked_patients’_risk_of_falling'].rank(pct=True) * 100).round()

all_merge['qm_depression_ptile'] = \
(all_merge.groupby('state')['how_often_the_home_health_team_checked_patients_for_depression'].rank(pct=True) * 100).round()

all_merge['qm_flu_shot_ptile'] = \
(all_merge.groupby('state')['how_often_the_home_health_team_determined_whether_patients_received_a_flu_shot_for_the_currnet_flu_season'].rank(pct=True) * 100).round()

all_merge['qm_pneum_shot_ptile'] = \
(all_merge.groupby('state')['how_often_the_home_health_team_made_sure_that_their_patients_have_received_a_pneumococcal_vaccine_pneumonia_shot'].rank(pct=True) * 100).round()

all_merge['qm_diabetes_footcare_ptile'] = \
(all_merge.groupby('state')['with_diabetes_how_often_the_home_health_team_got_doctor’s_orders_gave_foot_care_and_taught_patients_about_foot_care'].rank(pct=True) * 100).round()

all_merge['qm_improv_mobil_ptile'] = \
(all_merge.groupby('state')['how_often_patients_got_better_at_walking_or_moving_around'].rank(pct=True) * 100).round()

all_merge['qm_improv_trans_ptile'] = \
(all_merge.groupby('state')['how_often_patients_got_better_at_getting_in_and_out_of_bed'].rank(pct=True) * 100).round()

all_merge['qm_bathing_ptile'] = \
(all_merge.groupby('state')['how_often_patients_got_better_at_bathing'].rank(pct=True) * 100).round()

all_merge['qm_pain_moving_ptile'] = \
(all_merge.groupby('state')['how_often_patients_had_less_pain_when_moving_around'].rank(pct=True) * 100).round()

all_merge['qm_breathing_ptile'] = \
(all_merge.groupby('state')['how_often_patients’_breathing_improved'].rank(pct=True) * 100).round()

all_merge['qm_improv_wounds_ptile'] = \
(all_merge.groupby('state')['how_often_patients’_wounds_improved_or_healed_after_an_operation'].rank(pct=True) * 100).round()

all_merge['qm_correct_meds_ptile'] = \
(all_merge.groupby('state')['how_often_patients_got_better_at_taking_their_drugs_correctly_by_mouth'].rank(pct=True) * 100).round()

all_merge['qm_admit_hosp_ptile'] = \
(all_merge.groupby('state')['how_often_home_health_patients_had_to_be_admitted_to_the_hospital'].rank(pct=True) * 100).round()

all_merge['qm_urgent_care_no_hosp_ptile'] = \
(all_merge.groupby('state')['how_often_patients_receiving_home_health_care_needed_urgent_unplanned_care_in_the_er_without_being_admitted'].rank(pct=True) * 100).round()

all_merge['hcaps_prof_care_ptile'] = \
(all_merge.groupby('state')['percent_of_patients_who_reported_that_their_home_health_team_gave_care_in_a_professional_way'].rank(pct=True) * 100).round()

all_merge['hcaps_communicated_ptile'] = \
(all_merge.groupby('state')['percent_of_patients_who_reported_that_their_home_health_team_communicated_well_with_them'].rank(pct=True) * 100).round()

all_merge['hcaps_discuss_med_ptile'] = \
(all_merge.groupby('state')['percent_of_patients_who_reported_that_their_home_health_team_discussed_medicines_pain_and_home_safety_with_them'].rank(pct=True) * 100).round()

all_merge['hcaps_above_8_ptile'] = \
(all_merge.groupby('state')['percent_of_patients_who_gave_their_home_health_agency_a_rating_of_9_or_10_on_a_scale_from_0_lowest_to_10_highest'].rank(pct=True) * 100).round()

all_merge['hcaps_definitely_recommend_ptile'] = \
(all_merge.groupby('state')['percent_of_patients_who_reported_yes_they_would_definitely_recommend_the_home_health_agency_to_friends_and_family'].rank(pct=True) * 100).round()

all_merge['def_sum'] = all_merge.groupby('state')['def_binary'].transform('sum')
all_merge['alleg_sum'] = all_merge.groupby('state')['alleg_binary'].transform('sum')
all_merge['sub_alleg_sum'] = all_merge.groupby('state')['sub_alleg_bin'].transform('sum')
all_merge['rehosp_alleg_sum'] = all_merge.groupby('state')['rehosp_bin'].transform('sum')
all_merge['comp_bin_sum'] = all_merge.groupby('state')['comp_binary'].transform('sum')

all_merge['state_freq'] = all_merge.groupby('state')['state'].transform('count')

all_merge['def_state_mean'] = all_merge['def_sum'] / all_merge['state_freq']
all_merge['alleg_state_mean'] = all_merge['alleg_sum'] / all_merge['state_freq']
all_merge['sub_alleg_state_mean'] = all_merge['sub_alleg_sum'] / all_merge['state_freq']
all_merge['rehosp_alleg_state_mean'] = all_merge['rehosp_alleg_sum'] / all_merge['state_freq']
all_merge['comp_bin_state_mean'] = all_merge['comp_bin_sum'] / all_merge['state_freq']


div_cols = ['how_often_the_home_health_team_began_their_patients’_care_in_a_timely_manner',\
            'how_often_the_home_health_team_taught_patients_or_their_family_caregivers_about_their_drugs',\
            'how_often_the_home_health_team_checked_patients’_risk_of_falling',\
            'how_often_the_home_health_team_checked_patients_for_depression',\
            'how_often_the_home_health_team_determined_whether_patients_received_a_flu_shot_for_the_currnet_flu_season',\
            'how_often_the_home_health_team_made_sure_that_their_patients_have_received_a_pneumococcal_vaccine_pneumonia_shot',\
            'with_diabetes_how_often_the_home_health_team_got_doctor’s_orders_gave_foot_care_and_taught_patients_about_foot_care',\
            'how_often_patients_got_better_at_walking_or_moving_around',\
            'how_often_patients_got_better_at_getting_in_and_out_of_bed',\
            'how_often_patients_got_better_at_bathing',\
            'how_often_patients_had_less_pain_when_moving_around',\
            'how_often_patients’_breathing_improved',\
           'how_often_patients’_wounds_improved_or_healed_after_an_operation',\
           'how_often_patients_got_better_at_taking_their_drugs_correctly_by_mouth',\
           'how_often_home_health_patients_had_to_be_admitted_to_the_hospital',\
           'how_often_patients_receiving_home_health_care_needed_urgent_unplanned_care_in_the_er_without_being_admitted',\
           'percent_of_patients_who_reported_that_their_home_health_team_gave_care_in_a_professional_way',\
           'percent_of_patients_who_reported_that_their_home_health_team_communicated_well_with_them',\
           'percent_of_patients_who_reported_that_their_home_health_team_discussed_medicines_pain_and_home_safety_with_them',\
           'percent_of_patients_who_gave_their_home_health_agency_a_rating_of_9_or_10_on_a_scale_from_0_lowest_to_10_highest',\
           'percent_of_patients_who_reported_yes_they_would_definitely_recommend_the_home_health_agency_to_friends_and_family']

all_merge.loc[:, div_cols] = all_merge.loc[:,div_cols].div(100,axis=0)
all_merge = all_merge.loc[:, ~all_merge.columns.duplicated()]

all_merge_big = all_merge.loc[all_merge['County_count'] >= 40]
all_merge_small = all_merge.loc[all_merge['County_count'] < 40]

#all_merge_big = all_merge

all_merge_big['qm_timely_manner_county_ptile'] = \
(all_merge_big.groupby('County_ID')['how_often_the_home_health_team_began_their_patients’_care_in_a_timely_manner'].rank(pct=True)* 100).round()

all_merge_big['qm_teach_drugs_county_ptile'] = \
(all_merge_big.groupby('County_ID')['how_often_the_home_health_team_taught_patients_or_their_family_caregivers_about_their_drugs'].rank(pct=True) * 100).round()

all_merge_big['qm_risk_falling_county_ptile'] = \
(all_merge_big.groupby('County_ID')['how_often_the_home_health_team_checked_patients’_risk_of_falling'].rank(pct=True) * 100).round()

all_merge_big['qm_depression_county_ptile'] = \
(all_merge_big.groupby('County_ID')['how_often_the_home_health_team_checked_patients_for_depression'].rank(pct=True) * 100).round()

all_merge_big['qm_flu_shot_county_ptile'] = \
(all_merge_big.groupby('County_ID')['how_often_the_home_health_team_determined_whether_patients_received_a_flu_shot_for_the_currnet_flu_season'].rank(pct=True) * 100).round()

all_merge_big['qm_pneum_shot_county_ptile'] = \
(all_merge_big.groupby('County_ID')['how_often_the_home_health_team_made_sure_that_their_patients_have_received_a_pneumococcal_vaccine_pneumonia_shot'].rank(pct=True) * 100).round()

all_merge_big['qm_diabetes_footcare_county_ptile'] = \
(all_merge_big.groupby('County_ID')['with_diabetes_how_often_the_home_health_team_got_doctor’s_orders_gave_foot_care_and_taught_patients_about_foot_care'].rank(pct=True) * 100).round()

all_merge_big['qm_improv_mobil_county_ptile'] = \
(all_merge_big.groupby('County_ID')['how_often_patients_got_better_at_walking_or_moving_around'].rank(pct=True) * 100).round()

all_merge_big['qm_improv_trans_county_ptile'] = \
(all_merge_big.groupby('County_ID')['how_often_patients_got_better_at_getting_in_and_out_of_bed'].rank(pct=True) * 100).round()

all_merge_big['qm_bathing_county_ptile'] = \
(all_merge_big.groupby('County_ID')['how_often_patients_got_better_at_bathing'].rank(pct=True) * 100).round()

all_merge_big['qm_pain_moving_county_ptile'] = \
(all_merge_big.groupby('County_ID')['how_often_patients_had_less_pain_when_moving_around'].rank(pct=True) * 100).round()

all_merge_big['qm_urgent_care_norehosp_county_ptile'] = \
(all_merge_big.groupby('County_ID')['how_often_patients_receiving_home_health_care_needed_urgent_unplanned_care_in_the_er_without_being_admitted'].rank(pct=True) * 100).round()

all_merge_big['qm_breathing_county_ptile'] = \
(all_merge_big.groupby('County_ID')['how_often_patients’_breathing_improved'].rank(pct=True) * 100).round()

all_merge_big['qm_improv_wounds_county_ptile'] = \
(all_merge_big.groupby('County_ID')['how_often_patients’_wounds_improved_or_healed_after_an_operation'].rank(pct=True) * 100).round()

all_merge_big['qm_correct_meds_county_ptile'] = \
(all_merge_big.groupby('County_ID')['how_often_patients_got_better_at_taking_their_drugs_correctly_by_mouth'].rank(pct=True) * 100).round()

all_merge_big['qm_admit_hosp_county_ptile'] = \
(all_merge_big.groupby('County_ID')['how_often_home_health_patients_had_to_be_admitted_to_the_hospital'].rank(pct=True) * 100).round()

all_merge_big['hcaps_prof_care_county_ptile'] = \
(all_merge_big.groupby('County_ID')['percent_of_patients_who_reported_that_their_home_health_team_gave_care_in_a_professional_way'].rank(pct=True) * 100).round()

all_merge_big['hcaps_communicated_county_ptile'] = \
(all_merge_big.groupby('County_ID')['percent_of_patients_who_reported_that_their_home_health_team_communicated_well_with_them'].rank(pct=True) * 100).round()

all_merge_big['hcaps_discuss_med_county_ptile'] = \
(all_merge_big.groupby('County_ID')['percent_of_patients_who_reported_that_their_home_health_team_discussed_medicines_pain_and_home_safety_with_them'].rank(pct=True) * 100).round()

all_merge_big['hcaps_above_8_county_ptile'] = \
(all_merge_big.groupby('County_ID')['percent_of_patients_who_gave_their_home_health_agency_a_rating_of_9_or_10_on_a_scale_from_0_lowest_to_10_highest'].rank(pct=True) * 100).round()

all_merge_big['hcaps_definitely_recommend_county_ptile'] = \
(all_merge_big.groupby('County_ID')['percent_of_patients_who_reported_yes_they_would_definitely_recommend_the_home_health_agency_to_friends_and_family'].rank(pct=True) * 100).round()


all_merge_big_small = pd.concat([all_merge_big,all_merge_small], axis=0, ignore_index=True, sort=True)

#all_merge_big_small = all_merge_big

all_merge_big_small.drop_duplicates(subset = ['County_ID','cms_certification_number_ccn'], inplace=True)

all_merge_big_small['pred_all_diff_ptile'] = \
all_merge_big_small.groupby('state')['pred_all_diff'].rank(pct=True) * 100

all_merge_big_small['pred_med_diff_ptile'] =\
all_merge_big_small.groupby('state')['pred_med_diff'].rank(pct=True) * 100

all_merge_big_small['pred_chf_diff_ptile'] =\
all_merge_big_small.groupby('state')['pred_chf_diff'].rank(pct=True) * 100

all_merge_big_small['pred_surg_diff_ptile'] =\
all_merge_big_small.groupby('state')['pred_surg_diff'].rank(pct=True) * 100

all_merge_big_small['pred_med_high_risk_diff_ptile'] =\
all_merge_big_small.groupby('state')['pred_med_high_diff'].rank(pct=True) * 100

all_merge_big_small['pred_med_low_risk_diff_ptile'] =\
all_merge_big_small.groupby('state')['pred_med_low_diff'].rank(pct=True) * 100

all_merge_big_small['pred_surg_high_risk_diff_ptile'] =\
all_merge_big_small.groupby('state')['pred_surg_high_diff'].rank(pct=True) * 100

all_merge_big_small['pred_surg_low_risk_diff_ptile'] =\
all_merge_big_small.groupby('state')['pred_surg_low_diff'].rank(pct=True) * 100

all_merge_big_small['pred_lung_diff_ptile'] =\
all_merge_big_small.groupby('state')['pred_lung_diff'].rank(pct=True) * 100

all_merge_big_small['qm_urgent_care_no_hosp_rank'] = 'N/A'
all_merge_big_small.loc[all_merge_big_small['qm_urgent_care_norehosp_county_ptile'] < 10,'qm_urgent_care_no_hosp_rank'] = 'A'
all_merge_big_small.loc[(all_merge_big_small['qm_urgent_care_norehosp_county_ptile'] >= 10) & (all_merge_big_small['qm_urgent_care_norehosp_county_ptile'] < 25),'qm_urgent_care_no_hosp_rank'] = 'B'
all_merge_big_small.loc[(all_merge_big_small['qm_urgent_care_norehosp_county_ptile'] >= 25) & (all_merge_big_small['qm_urgent_care_norehosp_county_ptile'] < 75),'qm_urgent_care_no_hosp_rank'] = 'C'
all_merge_big_small.loc[(all_merge_big_small['qm_urgent_care_norehosp_county_ptile'] >= 75) & (all_merge_big_small['qm_urgent_care_norehosp_county_ptile'] < 90),'qm_urgent_care_no_hosp_rank'] = 'D'
all_merge_big_small.loc[(all_merge_big_small['qm_urgent_care_norehosp_county_ptile'] >= 90) & (all_merge_big_small['qm_urgent_care_norehosp_county_ptile'] < 100),'qm_urgent_care_no_hosp_rank'] = 'E'

all_merge_big_small['qm_hosp_admit_ptile_rank'] = 'N/A'
all_merge_big_small.loc[all_merge_big_small['qm_admit_hosp_county_ptile'] < 10,'qm_admit_hosp_county_rank'] = 'A'
all_merge_big_small.loc[(all_merge_big_small['qm_admit_hosp_county_ptile'] >= 10) & (all_merge_big_small['qm_admit_hosp_county_ptile'] < 25),'qm_hosp_admit_ptile_rank'] = 'B'
all_merge_big_small.loc[(all_merge_big_small['qm_admit_hosp_county_ptile'] >= 25) & (all_merge_big_small['qm_admit_hosp_county_ptile'] < 75),'qm_hosp_admit_ptile_rank'] = 'C'
all_merge_big_small.loc[(all_merge_big_small['qm_admit_hosp_county_ptile'] >= 75) & (all_merge_big_small['qm_admit_hosp_county_ptile'] < 90),'qm_hosp_admit_ptile_rank'] = 'D'
all_merge_big_small.loc[(all_merge_big_small['qm_admit_hosp_county_ptile'] >= 90) & (all_merge_big_small['qm_admit_hosp_county_ptile'] < 100),'qm_hosp_admit_ptile_rank'] = 'E'

all_merge_big_small['pred_all_diff_ptile_rank'] = 'N/A'
all_merge_big_small.loc[all_merge_big_small['pred_all_diff_ptile'] < 10,'pred_all_diff_ptile_rank'] = 'A'
all_merge_big_small.loc[(all_merge_big_small['pred_all_diff_ptile'] >= 10) & (all_merge_big_small['pred_all_diff_ptile'] < 25),'pred_all_diff_ptile_rank'] = 'B'
all_merge_big_small.loc[(all_merge_big_small['pred_all_diff_ptile'] >= 25) & (all_merge_big_small['pred_all_diff_ptile'] < 75),'pred_all_diff_ptile_rank'] = 'C'
all_merge_big_small.loc[(all_merge_big_small['pred_all_diff_ptile'] >= 75) & (all_merge_big_small['pred_all_diff_ptile'] < 90),'pred_all_diff_ptile_rank'] = 'D'
all_merge_big_small.loc[(all_merge_big_small['pred_all_diff_ptile'] >= 90) & (all_merge_big_small['pred_all_diff_ptile'] < 100),'pred_all_diff_ptile_rank'] = 'E'

all_merge_big_small.loc[(all_merge_big_small['pred_all_small_flag'] == 1) & (all_merge_big_small['pred_all_diff_ptile_rank'] == 'E'),'pred_all_diff_ptile_rank'] = 'D'
all_merge_big_small.loc[(all_merge_big_small['pred_all_small_flag'] == 1) & (all_merge_big_small['pred_all_diff_ptile_rank'] == 'A'),'pred_all_diff_ptile_rank'] = 'B'

all_merge_big_small['pred_med_diff_ptile_rank'] = 'N/A'
all_merge_big_small.loc[all_merge_big_small['pred_med_diff_ptile'] < 10,'pred_med_diff_ptile_rank'] = 'A'
all_merge_big_small.loc[(all_merge_big_small['pred_med_diff_ptile'] >= 10) & (all_merge_big_small['pred_med_diff_ptile'] < 25),'pred_med_diff_ptile_rank'] = 'B'
all_merge_big_small.loc[(all_merge_big_small['pred_med_diff_ptile'] >= 25) & (all_merge_big_small['pred_med_diff_ptile'] < 75),'pred_med_diff_ptile_rank'] = 'C'
all_merge_big_small.loc[(all_merge_big_small['pred_med_diff_ptile'] >= 75) & (all_merge_big_small['pred_med_diff_ptile'] < 90),'pred_med_diff_ptile_rank'] = 'D'
all_merge_big_small.loc[(all_merge_big_small['pred_med_diff_ptile'] >= 90) & (all_merge_big_small['pred_med_diff_ptile'] < 100),'pred_med_diff_ptile_rank'] = 'E'

all_merge_big_small.loc[(all_merge_big_small['pred_med_other_small_flag'] == 1) & (all_merge_big_small['pred_med_diff_ptile_rank'] == 'E'),'pred_med_diff_ptile_rank'] = 'D'
all_merge_big_small.loc[(all_merge_big_small['pred_med_other_small_flag'] == 1) & (all_merge_big_small['pred_med_diff_ptile_rank'] == 'A'),'pred_med_diff_ptile_rank'] = 'B'

all_merge_big_small['pred_chf_diff_ptile_rank'] = 'N/A'
all_merge_big_small.loc[all_merge_big_small['pred_chf_diff_ptile'] < 10,'pred_chf_diff_ptile_rank'] = 'A'
all_merge_big_small.loc[(all_merge_big_small['pred_chf_diff_ptile'] >= 10) & (all_merge_big_small['pred_chf_diff_ptile'] < 25),'pred_chf_diff_ptile_rank'] = 'B'
all_merge_big_small.loc[(all_merge_big_small['pred_chf_diff_ptile'] >= 25) & (all_merge_big_small['pred_chf_diff_ptile'] < 75),'pred_chf_diff_ptile_rank'] = 'C'
all_merge_big_small.loc[(all_merge_big_small['pred_chf_diff_ptile'] >= 75) & (all_merge_big_small['pred_chf_diff_ptile'] < 90),'pred_chf_diff_ptile_rank'] = 'D'
all_merge_big_small.loc[(all_merge_big_small['pred_chf_diff_ptile'] >= 90) & (all_merge_big_small['pred_chf_diff_ptile'] < 100),'pred_chf_diff_ptile_rank'] = 'E'

all_merge_big_small.loc[(all_merge_big_small['pred_CHF_small_flag'] == 1) & (all_merge_big_small['pred_chf_diff_ptile_rank'] == 'E'),'pred_chf_diff_ptile_rank'] = 'D'
all_merge_big_small.loc[(all_merge_big_small['pred_CHF_small_flag'] == 1) & (all_merge_big_small['pred_chf_diff_ptile_rank'] == 'A'),'pred_chf_diff_ptile_rank'] = 'B'

all_merge_big_small['pred_med_high_diff_ptile_rank'] = 'N/A'
all_merge_big_small.loc[all_merge_big_small['pred_med_high_risk_diff_ptile'] < 10,'pred_med_high_diff_ptile_rank'] = 'A'
all_merge_big_small.loc[(all_merge_big_small['pred_med_high_risk_diff_ptile'] >= 10) & (all_merge_big_small['pred_med_high_risk_diff_ptile'] < 25),'pred_med_high_diff_ptile_rank'] = 'B'
all_merge_big_small.loc[(all_merge_big_small['pred_med_high_risk_diff_ptile'] >= 25) & (all_merge_big_small['pred_med_high_risk_diff_ptile'] < 75),'pred_med_high_diff_ptile_rank'] = 'C'
all_merge_big_small.loc[(all_merge_big_small['pred_med_high_risk_diff_ptile'] >= 75) & (all_merge_big_small['pred_med_high_risk_diff_ptile'] < 90),'pred_med_high_diff_ptile_rank'] = 'D'
all_merge_big_small.loc[(all_merge_big_small['pred_med_high_risk_diff_ptile'] >= 90) & (all_merge_big_small['pred_med_high_risk_diff_ptile'] < 100),'pred_med_high_diff_ptile_rank'] = 'E'

all_merge_big_small.loc[(all_merge_big_small['pred_med_high_risk_small_flag'] == 1) & (all_merge_big_small['pred_med_high_diff_ptile_rank'] == 'E'),'pred_med_high_diff_ptile_rank'] = 'D'
all_merge_big_small.loc[(all_merge_big_small['pred_med_high_risk_small_flag'] == 1) & (all_merge_big_small['pred_med_high_diff_ptile_rank'] == 'A'),'pred_med_high_diff_ptile_rank'] = 'B'

all_merge_big_small['pred_med_low_diff_ptile_rank'] = 'N/A'
all_merge_big_small.loc[all_merge_big_small['pred_med_low_risk_diff_ptile'] < 10,'pred_med_low_diff_ptile_rank'] = 'A'
all_merge_big_small.loc[(all_merge_big_small['pred_med_low_risk_diff_ptile'] >= 10) & (all_merge_big_small['pred_med_low_risk_diff_ptile'] < 25),'pred_med_low_diff_ptile_rank'] = 'B'
all_merge_big_small.loc[(all_merge_big_small['pred_med_low_risk_diff_ptile'] >= 25) & (all_merge_big_small['pred_med_low_risk_diff_ptile'] < 75),'pred_med_low_diff_ptile_rank'] = 'C'
all_merge_big_small.loc[(all_merge_big_small['pred_med_low_risk_diff_ptile'] >= 75) & (all_merge_big_small['pred_med_low_risk_diff_ptile'] < 90),'pred_med_low_diff_ptile_rank'] = 'D'
all_merge_big_small.loc[(all_merge_big_small['pred_med_low_risk_diff_ptile'] >= 90) & (all_merge_big_small['pred_med_low_risk_diff_ptile'] < 100),'pred_med_low_diff_ptile_rank'] = 'E'

all_merge_big_small.loc[(all_merge_big_small['pred_med_low_risk_small_flag'] == 1) & (all_merge_big_small['pred_med_low_diff_ptile_rank'] == 'E'),'pred_med_low_diff_ptile_rank'] = 'D'
all_merge_big_small.loc[(all_merge_big_small['pred_med_low_risk_small_flag'] == 1) & (all_merge_big_small['pred_med_low_diff_ptile_rank'] == 'A'),'pred_med_low_diff_ptile_rank'] = 'B'

all_merge_big_small['pred_surg_high_risk_diff_ptile_rank'] = 'N/A'
all_merge_big_small.loc[all_merge_big_small['pred_surg_high_risk_diff_ptile'] < 10,'pred_surg_high_risk_diff_ptile_rank'] = 'A'
all_merge_big_small.loc[(all_merge_big_small['pred_surg_high_risk_diff_ptile'] >= 10) & (all_merge_big_small['pred_surg_high_risk_diff_ptile'] < 25),'pred_surg_high_risk_diff_ptile_rank'] = 'B'
all_merge_big_small.loc[(all_merge_big_small['pred_surg_high_risk_diff_ptile'] >= 25) & (all_merge_big_small['pred_surg_high_risk_diff_ptile'] < 75),'pred_surg_high_risk_diff_ptile_rank'] = 'C'
all_merge_big_small.loc[(all_merge_big_small['pred_surg_high_risk_diff_ptile'] >= 75) & (all_merge_big_small['pred_surg_high_risk_diff_ptile'] < 90),'pred_surg_high_risk_diff_ptile_rank'] = 'D'
all_merge_big_small.loc[(all_merge_big_small['pred_surg_high_risk_diff_ptile'] >= 90) & (all_merge_big_small['pred_surg_high_risk_diff_ptile'] < 100),'pred_surg_high_risk_diff_ptile_rank'] = 'E'

all_merge_big_small.loc[(all_merge_big_small['pred_surg_high_risk_small_flag'] == 1) & (all_merge_big_small['pred_surg_high_risk_diff_ptile_rank'] == 'E'),'pred_surg_high_risk_diff_ptile_rank'] = 'D'
all_merge_big_small.loc[(all_merge_big_small['pred_surg_high_risk_small_flag'] == 1) & (all_merge_big_small['pred_surg_high_risk_diff_ptile_rank'] == 'A'),'pred_surg_high_risk_diff_ptile_rank'] = 'B'


all_merge_big_small['pred_surg_diff_ptile_rank'] = 'N/A'
all_merge_big_small.loc[all_merge_big_small['pred_surg_diff_ptile'] < 10,'pred_surg_diff_ptile_rank'] = 'A'
all_merge_big_small.loc[(all_merge_big_small['pred_surg_diff_ptile'] >= 10) & (all_merge_big_small['pred_surg_diff_ptile'] < 25),'pred_surg_diff_ptile_rank'] = 'B'
all_merge_big_small.loc[(all_merge_big_small['pred_surg_diff_ptile'] >= 25) & (all_merge_big_small['pred_surg_diff_ptile'] < 75),'pred_surg_diff_ptile_rank'] = 'C'
all_merge_big_small.loc[(all_merge_big_small['pred_surg_diff_ptile'] >= 75) & (all_merge_big_small['pred_surg_diff_ptile'] < 90),'pred_surg_diff_ptile_rank'] = 'D'
all_merge_big_small.loc[(all_merge_big_small['pred_surg_diff_ptile'] >= 90) & (all_merge_big_small['pred_surg_diff_ptile'] < 100),'pred_surg_diff_ptile_rank'] = 'E'

all_merge_big_small.loc[(all_merge_big_small['pred_surg_other_small_flag'] == 1) & (all_merge_big_small['pred_surg_diff_ptile_rank'] == 'E'),'pred_surg_diff_ptile_rank'] = 'D'
all_merge_big_small.loc[(all_merge_big_small['pred_surg_other_small_flag'] == 1) & (all_merge_big_small['pred_surg_diff_ptile_rank'] == 'A'),'pred_surg_diff_ptile_rank'] = 'B'

all_merge_big_small['pred_surg_low_risk_diff_ptile_rank'] = 'N/A'
all_merge_big_small.loc[all_merge_big_small['pred_surg_low_risk_diff_ptile'] < 10,'pred_surg_low_risk_diff_ptile_rank'] = 'A'
all_merge_big_small.loc[(all_merge_big_small['pred_surg_low_risk_diff_ptile'] >= 10) & (all_merge_big_small['pred_surg_low_risk_diff_ptile'] < 25),'pred_surg_low_risk_diff_ptile_rank'] = 'B'
all_merge_big_small.loc[(all_merge_big_small['pred_surg_low_risk_diff_ptile'] >= 25) & (all_merge_big_small['pred_surg_low_risk_diff_ptile'] < 75),'pred_surg_low_risk_diff_ptile_rank'] = 'C'
all_merge_big_small.loc[(all_merge_big_small['pred_surg_low_risk_diff_ptile'] >= 75) & (all_merge_big_small['pred_surg_low_risk_diff_ptile'] < 90),'pred_surg_low_risk_diff_ptile_rank'] = 'D'
all_merge_big_small.loc[(all_merge_big_small['pred_surg_low_risk_diff_ptile'] >= 90) & (all_merge_big_small['pred_surg_low_risk_diff_ptile'] < 100),'pred_surg_low_risk_diff_ptile_rank'] = 'E'

all_merge_big_small.loc[(all_merge_big_small['pred_surg_low_risk_small_flag'] == 1) & (all_merge_big_small['pred_surg_low_risk_diff_ptile_rank'] == 'E'),'pred_surg_low_risk_diff_ptile_rank'] = 'D'
all_merge_big_small.loc[(all_merge_big_small['pred_surg_low_risk_small_flag'] == 1) & (all_merge_big_small['pred_surg_low_risk_diff_ptile_rank'] == 'A'),'pred_surg_low_risk_diff_ptile_rank'] = 'B'

all_merge_big_small['pred_lung_diff_ptile_rank'] = 'N/A'
all_merge_big_small.loc[all_merge_big_small['pred_lung_diff_ptile'] < 10,'pred_lung_diff_ptile_rank'] = 'A'
all_merge_big_small.loc[(all_merge_big_small['pred_lung_diff_ptile'] >= 10) & (all_merge_big_small['pred_lung_diff_ptile'] < 25),'pred_lung_diff_ptile_rank'] = 'B'
all_merge_big_small.loc[(all_merge_big_small['pred_lung_diff_ptile'] >= 25) & (all_merge_big_small['pred_lung_diff_ptile'] < 75),'pred_lung_diff_ptile_rank'] = 'C'
all_merge_big_small.loc[(all_merge_big_small['pred_lung_diff_ptile'] >= 75) & (all_merge_big_small['pred_lung_diff_ptile'] < 90),'pred_lung_diff_ptile_rank'] = 'D'
all_merge_big_small.loc[(all_merge_big_small['pred_lung_diff_ptile'] >= 90) & (all_merge_big_small['pred_lung_diff_ptile'] < 100),'pred_lung_diff_ptile_rank'] = 'E'

all_merge_big_small.loc[(all_merge_big_small['pred_lung_small_flag'] == 1) & (all_merge_big_small['pred_lung_diff_ptile_rank'] == 'E'),'pred_lung_diff_ptile_rank'] = 'D'
all_merge_big_small.loc[(all_merge_big_small['pred_lung_small_flag'] == 1) & (all_merge_big_small['pred_lung_diff_ptile_rank'] == 'A'),'pred_lung_diff_ptile_rank'] = 'B'

all_merge_big_small['oe_all_mean'] = all_merge_big_small.groupby(['pred_all_diff_ptile_rank'])['oe_all_estimate'].transform('mean')
all_merge_big_small['oe_med_other_mean'] = all_merge_big_small.groupby(['pred_med_diff_ptile_rank'])['oe_med_estimate'].transform('mean')
all_merge_big_small['oe_surg_other_mean'] = all_merge_big_small.groupby(['pred_surg_diff_ptile_rank'])['oe_surg_estimate'].transform('mean')
all_merge_big_small['oe_chf_mean'] = all_merge_big_small.groupby(['pred_chf_diff_ptile_rank'])['oe_chf_estimate'].transform('mean')
all_merge_big_small['oe_med_low_mean'] = all_merge_big_small.groupby(['pred_med_low_diff_ptile_rank'])['oe_med_low_estimate'].transform('mean')
all_merge_big_small['oe_med_high_mean'] = all_merge_big_small.groupby(['pred_med_high_diff_ptile_rank'])['oe_med_high_estimate'].transform('mean')
all_merge_big_small['oe_surg_high_mean'] = all_merge_big_small.groupby(['pred_surg_high_risk_diff_ptile_rank'])['oe_surg_high_estimate'].transform('mean')
all_merge_big_small['oe_surg_low_mean'] = all_merge_big_small.groupby(['pred_surg_low_risk_diff_ptile_rank'])['oe_surg_low_estimate'].transform('mean')
all_merge_big_small['oe_lung_mean'] = all_merge_big_small.groupby(['pred_lung_diff_ptile_rank'])['oe_lung_estimate'].transform('mean')

all_merge_big_small['HC_prof_care_high_mean'] = all_merge_big_small.groupby(['County_ID'])['HC_prof_care_high_binary'].transform('mean')
all_merge_big_small['HC_prof_care_low_mean'] = all_merge_big_small.groupby(['County_ID'])['HC_prof_care_low_binary'].transform('mean')
all_merge_big_small['HC_good_comm_high_mean'] = all_merge_big_small.groupby(['County_ID'])['HC_good_comm_high_binary'].transform('mean')
all_merge_big_small['HC_good_comm_low_mean'] = all_merge_big_small.groupby(['County_ID'])['HC_good_comm_low_binary'].transform('mean')
all_merge_big_small['HC_dis_med_pain_saf_high_mean'] = all_merge_big_small.groupby(['County_ID'])['HC_dis_med_pain_saf_low_binary'].transform('mean')
all_merge_big_small['HC_dis_med_pain_saf_low_mean'] = all_merge_big_small.groupby(['County_ID'])['HC_dis_med_pain_saf_low_binary'].transform('mean')
all_merge_big_small['HC_hh_rate_over8_high_mean'] = all_merge_big_small.groupby(['County_ID'])['HC_hh_rate_over8_high_binary'].transform('mean')
all_merge_big_small['HC_hh_rate_over8_high_mean'] = all_merge_big_small.groupby(['County_ID'])['HC_hh_rate_over8_high_binary'].transform('mean')
all_merge_big_small['HC_def_rec_high_mean'] = all_merge_big_small.groupby(['County_ID'])['HC_def_rec_high_binary'].transform('mean')
all_merge_big_small['HC_def_rec_low_mean'] = all_merge_big_small.groupby(['County_ID'])['HC_def_rec_low_binary'].transform('mean')

all_merge_big_small['def_sum_county'] = all_merge_big_small.groupby('County_ID')['def_binary'].transform('sum')
all_merge_big_small['alleg_sum_county'] = all_merge_big_small.groupby('County_ID')['alleg_binary'].transform('sum')
all_merge_big_small['sub_alleg_sum_county'] = all_merge_big_small.groupby('County_ID')['sub_alleg_bin'].transform('sum')
all_merge_big_small['rehosp_alleg_sum_county'] = all_merge_big_small.groupby('County_ID')['rehosp_bin'].transform('sum')
all_merge_big_small['comp_bin_sum_county'] = all_merge_big_small.groupby('County_ID')['comp_binary'].transform('sum')

all_merge_big_small['CountyID_freq'] = all_merge_big_small.groupby('County_ID')['County_ID'].transform('count')

all_merge_big_small['def_county_mean'] = all_merge_big_small['def_sum_county'] / all_merge_big_small['CountyID_freq']
all_merge_big_small['alleg_county_mean'] = all_merge_big_small['alleg_sum_county'] / all_merge_big_small['CountyID_freq']
all_merge_big_small['sub_alleg_county_mean'] = all_merge_big_small['sub_alleg_sum_county'] / all_merge_big_small['CountyID_freq']
all_merge_big_small['rehosp_alleg_county_mean'] = all_merge_big_small['rehosp_alleg_sum_county'] / all_merge_big_small['CountyID_freq']
all_merge_big_small['comp_bin_county_mean'] = all_merge_big_small['comp_bin_sum_county'] / all_merge_big_small['CountyID_freq']

all_merge_big_small['Ownership_Type'] = 'N/A'
all_merge_big_small.loc[all_merge_big_small['type_of_ownership_x'].str.contains("Non - Profit",na=False),'Ownership_Type'] = "Non-Profit"
all_merge_big_small.loc[all_merge_big_small['type_of_ownership_x'].str.contains("Government",na=False),'Ownership_Type'] = "Government"
all_merge_big_small.loc[all_merge_big_small['type_of_ownership_x'].str.contains("Proprietary",na=False),'Ownership_Type'] = "For-Profit"

all_merge_big_small['Lab_Service_Binary'] = np.where(all_merge_big_small['LAB_SRVC_DESC'] == 'NOT PROVIDED', 'No', 'Yes')
all_merge_big_small['Pharm_Service_Binary'] = np.where(all_merge_big_small['PHRMCY_SRVC_DESC'] == 'NOT PROVIDED','No', 'Yes')
all_merge_big_small['Nursing_Service_Binary'] = np.where(all_merge_big_small['NRSNG_SRVC_DESC'] == 'NOT PROVIDED', 'No', 'Yes')
all_merge_big_small['PT_Service_Binary'] = np.where(all_merge_big_small['PT_SRVC_DESC'] == 'NOT PROVIDED', 'No', 'Yes')
all_merge_big_small['OT_Service_Binary'] = np.where(all_merge_big_small['OCPTNL_THRPST_SRVC_DESC'] == 'NOT PROVIDED', 'No', 'Yes')
all_merge_big_small['Speech_Service_Binary'] = np.where(all_merge_big_small['SPCH_THRPY_SRVC_DESC'] == 'NOT PROVIDED', 'No', 'Yes')
all_merge_big_small['Med_SS_Service_Binary'] = np.where(all_merge_big_small['MDCL_SCL_SRVC_DESC'] == 'NOT PROVIDED', 'No', 'Yes')
all_merge_big_small['HH_Aide_Service_Binary'] = np.where(all_merge_big_small['HH_AIDE_SRVC_DESC'] == 'NOT PROVIDED', 'No', 'Yes')

all_merge_big_small['qm_pac_urgent_care_no_hosp_rank'] = 'N/A'
all_merge_big_small.loc[(all_merge_big_small['how_often_home_health_patients_who_have_had_a_recent_hospital_stay_received_care_in_the_hospital_emergency_room_without_being_readmitted_to_the_hospital'] == 'Worse Than Expected') ,'qm_pac_urgent_care_no_hosp_rank'] = 'D'
all_merge_big_small.loc[(all_merge_big_small['how_often_home_health_patients_who_have_had_a_recent_hospital_stay_received_care_in_the_hospital_emergency_room_without_being_readmitted_to_the_hospital'] == 'Same As Expected') ,'qm_pac_urgent_care_no_hosp_rank'] = 'C'
all_merge_big_small.loc[(all_merge_big_small['how_often_home_health_patients_who_have_had_a_recent_hospital_stay_received_care_in_the_hospital_emergency_room_without_being_readmitted_to_the_hospital'] == 'Better Than Expected') ,'qm_pac_urgent_care_no_hosp_rank'] = 'A'

all_merge_big_small['qm_pac_hosp_admitt_rank'] = 'N/A'
all_merge_big_small.loc[(all_merge_big_small['how_often_home_health_patients_who_have_had_a_recent_hospital_stay_had_to_be_re-admitted_to_the_hospital'] == 'Worse Than Expected') ,'qm_pac_hosp_admitt_rank'] = 'D'
all_merge_big_small.loc[(all_merge_big_small['how_often_home_health_patients_who_have_had_a_recent_hospital_stay_had_to_be_re-admitted_to_the_hospital'] == 'Same As Expected') ,'qm_pac_hosp_admitt_rank'] = 'C'
all_merge_big_small.loc[(all_merge_big_small['how_often_home_health_patients_who_have_had_a_recent_hospital_stay_had_to_be_re-admitted_to_the_hospital'] == 'Better Than Expected') ,'qm_pac_hosp_admitt_rank'] = 'A'

all_merge_big_small['pred_all_comp_rate'] = all_merge_big_small['oe_all_mean'] * 0.147237
all_merge_big_small['pred_all_state_rate'] = all_merge_big_small['oe_all_state_median'] * 0.147237
all_merge_big_small['pred_med_other_comp_rate'] = all_merge_big_small['oe_med_other_mean'] * 0.175482
all_merge_big_small['pred_med_other_state_rate'] = all_merge_big_small['oe_med_state_median'] * 0.175482
all_merge_big_small['pred_surg_other_comp_rate'] = all_merge_big_small['oe_surg_other_mean'] * 0.071103
all_merge_big_small['pred_surg_other_state_rate'] = all_merge_big_small['oe_surg_state_median'] * 0.071103
all_merge_big_small['pred_CHF_comp_rate'] = all_merge_big_small['oe_chf_mean'] * 0.175482
all_merge_big_small['pred_CHF_state_rate'] = all_merge_big_small['oe_chf_state_median'] * 0.175482
all_merge_big_small['pred_med_high_comp_rate'] = all_merge_big_small['oe_med_high_mean'] * 0.175482
all_merge_big_small['pred_med_high_state_rate'] = all_merge_big_small['oe_med_high_state_median'] * 0.175482
all_merge_big_small['pred_med_low_comp_rate'] = all_merge_big_small['oe_med_low_mean'] * 0.175482
all_merge_big_small['pred_med_low_state_rate'] = all_merge_big_small['oe_med_low_state_median'] * 0.175482
all_merge_big_small['pred_surg_high_comp_rate'] = all_merge_big_small['oe_surg_high_mean'] * 0.071103
all_merge_big_small['pred_surg_high_state_rate'] = all_merge_big_small['oe_surg_high_state_median'] * 0.071103
all_merge_big_small['pred_surg_low_comp_rate'] = all_merge_big_small['oe_surg_low_mean'] * 0.071103
all_merge_big_small['pred_surg_low_state_rate'] = all_merge_big_small['oe_surg_low_state_median'] * 0.071103
all_merge_big_small['pred_lung_diff_comp_rate'] = all_merge_big_small['oe_lung_mean'] * 0.071103
all_merge_big_small['pred_lung_diff_state_rate'] = all_merge_big_small['oe_lung_state_median'] * 0.175482

all_merge_big_small.rename(columns={'date_certified_x': 'date_certified', 'STATE_ABBREV': 'State_Srv', 'County_ID': 'Ecosystem_Name'}, inplace=True)

all_merge_big_small['CMS_ID'] = "'" + all_merge_big_small['cms_certification_number_ccn'].astype(str)
all_merge_big_small['address'] = all_merge_big_small['address_x'].str.replace(',',' ').str.upper()
all_merge_big_small['city'] = all_merge_big_small['city'].str.upper()
all_merge_big_small['state'] = all_merge_big_small['state'].str.upper()
all_merge_big_small['provider_name'] = all_merge_big_small['provider_name_x'].str.replace(',',' ').str.title()
all_merge_big_small['Reporting_Month'] = currentYear + '/' + currentMonth

all_merge_big_small = all_merge_big_small[pd.notnull(all_merge_big_small['state'])]

nacols = ['def_total', 'alleg_total','tagcount_lastsurv','tagcount_complaint']
nacolsoth = ['percent_of_patients_who_reported_that_their_home_health_team_discussed_medicines_pain_and_home_safety_with_them',
                                                              'percent_of_patients_who_reported_that_their_home_health_team_communicated_well_with_them',
                                                              'how_often_home_health_patients_had_to_be_admitted_to_the_hospital',
                                                              'quality_of_patient_care_star_rating']
all_merge_big_small[nacols] = all_merge_big_small[nacols].fillna(0)
all_merge_big_small[nacolsoth] = all_merge_big_small[nacolsoth].fillna('N/A')


all_merge_big_small = all_merge_big_small[all_merge_big_small.state != 100][['CMS_ID', 'Ecosystem_Name', 'provider_name', 'address', 'city',
                                                               'state', 'zip_code', 'county', 'County_count', 'State_Srv', 'Ownership_Type', 'Lab_Service_Binary',
                                                               'Pharm_Service_Binary', 'Nursing_Service_Binary', 'PT_Service_Binary',
                                                               'OT_Service_Binary', 'Speech_Service_Binary','Med_SS_Service_Binary', 'HH_Aide_Service_Binary',
                                                               'date_certified','quality_of_patient_care_star_rating','hhcahps_survey_summary_star_rating',
                                                              'how_often_the_home_health_team_began_their_patients’_care_in_a_timely_manner',
                                                              'qm_timely_manner_county_ptile','qm_timely_manner_ptile',
                                                              'how_often_the_home_health_team_taught_patients_or_their_family_caregivers_about_their_drugs',
                                                              'qm_teach_drugs_county_ptile','qm_teach_drugs_ptile',
                                                              'how_often_the_home_health_team_checked_patients’_risk_of_falling',
                                                              'qm_risk_falling_county_ptile','qm_risk_falling_ptile',
                                                              'how_often_the_home_health_team_checked_patients_for_depression',
                                                              'qm_depression_county_ptile', 'qm_depression_ptile',
                                                              'how_often_the_home_health_team_determined_whether_patients_received_a_flu_shot_for_the_currnet_flu_season',
                                                              'qm_flu_shot_county_ptile','qm_flu_shot_ptile',
                                                              'how_often_the_home_health_team_made_sure_that_their_patients_have_received_a_pneumococcal_vaccine_pneumonia_shot',
                                                              'qm_pneum_shot_county_ptile','qm_pneum_shot_ptile',
                                                              'with_diabetes_how_often_the_home_health_team_got_doctor’s_orders_gave_foot_care_and_taught_patients_about_foot_care',
                                                              'qm_diabetes_footcare_county_ptile','qm_diabetes_footcare_ptile',
                                                              'how_often_patients_got_better_at_walking_or_moving_around',
                                                              'qm_improv_mobil_county_ptile','qm_improv_mobil_ptile',
                                                              'how_often_patients_got_better_at_getting_in_and_out_of_bed',
                                                              'qm_improv_trans_county_ptile', 'qm_improv_trans_ptile',
                                                              'how_often_patients_got_better_at_bathing', 'qm_bathing_county_ptile',
                                                              'qm_bathing_ptile','how_often_patients_had_less_pain_when_moving_around',
                                                              'qm_pain_moving_county_ptile', 'qm_pain_moving_ptile',
                                                              'how_often_patients’_breathing_improved','qm_breathing_county_ptile',
                                                              'qm_breathing_ptile','how_often_patients’_wounds_improved_or_healed_after_an_operation',
                                                              'qm_improv_wounds_county_ptile','qm_improv_wounds_ptile',
                                                              'how_often_patients_got_better_at_taking_their_drugs_correctly_by_mouth',
                                                              'qm_correct_meds_county_ptile','qm_correct_meds_ptile',
                                                              'how_often_home_health_patients_had_to_be_admitted_to_the_hospital','qm_admit_hosp_ptile',
                                                              'qm_hosp_admit_ptile_rank','how_often_patients_receiving_home_health_care_needed_urgent_unplanned_care_in_the_er_without_being_admitted','qm_urgent_care_norehosp_county_ptile',
                                                              'qm_urgent_care_no_hosp_ptile','qm_urgent_care_no_hosp_rank',
                                                              'how_often_home_health_patients_who_have_had_a_recent_hospital_stay_had_to_be_re-admitted_to_the_hospital',
                                                              'qm_pac_hosp_admitt_rank', 'how_often_home_health_patients_who_have_had_a_recent_hospital_stay_received_care_in_the_hospital_emergency_room_without_being_readmitted_to_the_hospital',
                                                              'qm_pac_urgent_care_no_hosp_rank','pred_all_diff_ptile_rank',
                                                              'pred_med_diff_ptile_rank','pred_chf_diff_ptile_rank','pred_med_high_diff_ptile_rank','pred_med_low_diff_ptile_rank','pred_surg_high_risk_diff_ptile_rank',
                                                              'pred_surg_diff_ptile_rank','pred_surg_low_risk_diff_ptile_rank','pred_lung_diff_ptile_rank','pred_all_comp_rate','pred_all_state_rate',
                                                              'pred_med_other_comp_rate','pred_med_other_state_rate','pred_surg_other_comp_rate','pred_surg_other_state_rate',
                                                              'pred_CHF_comp_rate','pred_CHF_state_rate','pred_med_high_comp_rate','pred_med_high_state_rate',
                                                              'pred_med_low_comp_rate','pred_med_low_state_rate','pred_surg_high_comp_rate','pred_surg_high_state_rate',
                                                              'pred_surg_low_comp_rate','pred_surg_low_state_rate','pred_lung_diff_comp_rate','pred_lung_diff_state_rate',
                                                              'percent_of_patients_who_reported_that_their_home_health_team_gave_care_in_a_professional_way',
                                                              'hcaps_prof_care_county_ptile', 'hcaps_prof_care_ptile', 'percent_of_patients_who_reported_that_their_home_health_team_communicated_well_with_them',
                                                              'hcaps_communicated_county_ptile', 'hcaps_communicated_ptile', 'hcaps_definitely_recommend_county_ptile',
                                                              'hcaps_definitely_recommend_ptile','hcaps_discuss_med_county_ptile','hcaps_discuss_med_ptile','hcaps_prof_care_county_ptile',
                                                              'hcaps_prof_care_ptile', 'percent_of_patients_who_reported_that_their_home_health_team_discussed_medicines_pain_and_home_safety_with_them',
                                                              'percent_of_patients_who_gave_their_home_health_agency_a_rating_of_9_or_10_on_a_scale_from_0_lowest_to_10_highest',
                                                              'hcaps_above_8_county_ptile', 'hcaps_above_8_ptile',
                                                              'percent_of_patients_who_reported_yes_they_would_definitely_recommend_the_home_health_agency_to_friends_and_family',
                                                              'hcaps_definitely_recommend_county_ptile','hcaps_definitely_recommend_ptile',
                                                              'alleg_state_mean','alleg_total','rehosp_alleg_state_mean', 'sub_alleg_state_mean',
                                                              'sub_alleg_sum','total_weight_alleg','alleg_county_mean', 'total_sub_alleg', 'total_rehosp_alleg',
                                                              'sub_alleg_county_mean','rehosp_alleg_county_mean', 'def_county_mean', 'def_state_mean', 'def_total',
                                                              'comp_bin_county_mean','comp_bin_state_mean','number_of_completed_surveys', 'response_rate', 'CRTFCTN_DT',
                                                              'CRTFCTN_UPDT_DT', 'Reporting_Month','tagcount_lastsurv','tagcount_complaint']]



network_tab = all_merge_big_small[['provider_name', 'CMS_ID', 'county', 'Ownership_Type', 'address', 'city',
                                                              'state', 'zip_code',
                                                              'quality_of_patient_care_star_rating','hhcahps_survey_summary_star_rating',
                                                              'percent_of_patients_who_reported_that_their_home_health_team_discussed_medicines_pain_and_home_safety_with_them',
                                                              'percent_of_patients_who_reported_that_their_home_health_team_communicated_well_with_them',
                                                              'how_often_home_health_patients_had_to_be_admitted_to_the_hospital',
                                                              'alleg_total', 'def_total','pred_med_high_diff_ptile_rank', 'pred_med_high_comp_rate',
                                                              'pred_med_low_diff_ptile_rank', 'pred_med_low_comp_rate',
                                                              'pred_surg_high_risk_diff_ptile_rank', 'pred_surg_high_comp_rate',
                                                              'pred_surg_low_risk_diff_ptile_rank','pred_surg_low_comp_rate',
                                                              'pred_chf_diff_ptile_rank','pred_CHF_comp_rate',
                                                              'pred_lung_diff_ptile_rank','pred_lung_diff_comp_rate',
                                                              'pred_med_diff_ptile_rank','pred_med_other_comp_rate',
                                                              'pred_surg_diff_ptile_rank','pred_surg_other_comp_rate'
                                                              ]]


for state in all_merge_big_small['state'].unique():
  if any(a in state for a in ('CA', 'AZ', 'FL', 'MD', 'NV')):
    final_file_name = '\\{0}_HH_ScoreCard_Fac_Info.csv'.format(state)
    all_merge_big_small[all_merge_big_small['state'] == state].sort_values(['state','Ecosystem_Name','provider_name','CMS_ID']).to_csv(out_recent_dir + final_file_name,index=False)

for state in network_tab['state'].unique():
  if any(a in state for a in ('CA', 'AZ', 'FL', 'MD', 'NV')):
    final_file_name_network = '\\{0}_HH_ScoreCard_Network.csv'.format(state)
    network_tab[network_tab['state'] == state].sort_values(['state','county','provider_name','CMS_ID']).to_csv(out_recent_dir + final_file_name_network,index=False)

dt_rng.to_csv(out_recent_dir + '\\DateRanges.csv', index=None,header=True)
