'''
前處理
1. 處理遺漏值(先將所有問號轉成 0 )
2. 欄位篩選(ssid 欄位捨棄)
3. 欄位型態轉換(將 HEX、Mac 轉成 Int、type類別 轉成1~4)
4. 將 type 欄位另存一檔並從 RowData 移除
5. 資料正規畫(Normalization_MinMax)

------------------------
Modified by : Patrick Yan
Modified DateTime : 2022/10/14 15:49
Description ： 
    增添了兩條創建資料夾的代碼
    createFolder(path_save + "BigData_MinMax")
    createFolder(path_save + "SmallData_MinMax")
    注釋了#IP_column_list = loadColumns("IP.txt")，沒有ip_column
------------------------
Modified by : Patrick Yan
Modified DateTime : 2022/12/19 02:49
Description ： 
    新增了 String To Num
    將所有F轉為1, T轉換為2
------------------------
'''

import os
import datetime
import numpy as np
import pandas as pd
import re
import ipaddress
from collections import Counter

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

# load index from 'path' text file
def loadColumns(path):
    file = open(path, "r")
    index = [line.strip() for line in file]
    file.close()
    return index


# pattern_hex = '^0x([0-9a-fA-F]{2,30})$'
# pattern_IP = r'(25[0-5]|2[0-4]\d|[0-1]\d{2}|[1-9]?\d)\.(25[0-5]|2[0-4]\d|[0-1]\d{2}|[1-9]?\d)\.(25[0-5]|2[0-4]\d|[0-1]\d{2}|[1-9]?\d)\.(25[0-5]|2[0-4]\d|[0-1]\d{2}|[1-9]?\d)'
# pattern = r'([A-Fa-f0-9]{2}:){5}[A-Fa-f0-9]{2}'
#
# def mapping(contx):
#     if re.match(pattern_hex,contx) != None:
#         # print(hex2int(contx))
#         return HEX_to_INT(contx)
#     elif re.match(pattern_IP,contx) != None:
#         return IP_to_INT(contx)
#     elif re.match(pattern,contx) != None:
#         return MAC_to_INT(contx)
#     return contx

def HEX_to_INT(HEX):
    HEX_INT = int((HEX),0)
    # HEX_INT = int(float(HEX_str.replace("0x", ""), 0))  # if re.match(pattern_zero,nu.replace("0x", "")) != None else 0
    return HEX_INT

# def MAC_to_INT(MAC_str):
#      MAC_str = MAC_str.translate(":")
#      print(MAC_str)
#      MAC_INT = int(MAC_str, 0)
#      return MAC_INT

def MAC_to_INT(MAC):
    MAC_INT = int(re.sub(r':*','', MAC),16)
    return MAC_INT

def IP_to_INT(IP):
    IP_INT = int(ipaddress.ip_address(IP)) #struct.unpack("!I", socket.inet_aton(addr))[0]
    return IP_INT

import netaddr

def IP_to_int(ip):
    if netaddr.IPAddress(ip).version == 4:
        ip = int(netaddr.IPAddress(ip))
    elif netaddr.IPAddress(ip).version == 6:
        ip = int(netaddr.IPNetwork(ip).value)
    else:
        ip = ip
    return float(ip)


# Xnor = (X - Xmin) / (Xmax - Xmin)
def minmax(x, Xmax, Xmin):
    if Xmax == Xmin:
        return 0
    else:
        nor = float((x - Xmin) / (Xmax - Xmin))
    return float(nor)


# PATH
path_r_raw_dataset = 'Data/0_DataSet/'
path_r_index = 'TON_IOT_Label.txt'
path_save = 'Data/'

# Create Folder
createFolder(path_save + "01_Normalization")

counter_File = 0
column_list = loadColumns(path_r_index)
MAC_column_list = loadColumns("MAC.txt")
HEX_column_list = loadColumns("HEX.txt")
IP_column_list = loadColumns("IP.txt")


startTime = datetime.datetime.now()
print('==================== Start: ', startTime, '====================')

for i, file in enumerate(os.listdir(path_r_raw_dataset)):
    if file.endswith(".csv"):
        fileName = file.replace(".csv", '')
        print('\n', counter_File, ':\t', fileName)
        counter_File += 1
        
        # load raw_dataset
        df_data = pd.read_csv(path_r_raw_dataset + file, sep=',', low_memory=False, index_col=False)
        print("columns list length : ", len(column_list), " --> ", column_list)
        print(df_data)
        # df_data_Trn = pd.read_csv(path_r_raw_dataset + file, header=None, sep=',', low_memory=False, index_col=False)
        # df_data_Tst = pd.read_csv(path_r_raw_dataset + file, header=None, sep=',', low_memory=False, index_col=False)
        # print(df_data)

        # 將所有dash轉為0
        print(" ==================== - to 0 is START =================== ")
        size_columns_num = len(column_list) - 1
        print(size_columns_num)

        for i in range(size_columns_num):
            # df_data['frame.dlt'] = df_data['frame.dlt'].replace('-', 0)
            # df_data['data.len'] = df_data['data.len'].replace('-', 0)
            df_data[column_list[i]] = df_data[column_list[i]].replace('-', 0)
        print(df_data)
        print(" ==================== - to 0 is END =================== ")

        # 將所有F轉為1, T轉換為2,"^' '$" stands for exact match
        print(" ==================== F,T to 1,2 is START =================== ")
        for i in range(size_columns_num):
            df_data[column_list[i]] = df_data[column_list[i]].replace(r'^F$', 1, regex=True)
            df_data[column_list[i]] = df_data[column_list[i]].replace(r'^T$', 2, regex=True)
        print(df_data)
        print(" ==================== F,T to 1,2 is END =================== ")

        # # HEX to INT
        # print(" ==================== HEX to 0 is START =================== ")
        # HEX_size_columns_num = len(HEX_column_list)
        # print(HEX_column_list)
        # print(HEX_size_columns_num)

        # for i in range(HEX_size_columns_num):
        #     df_data[HEX_column_list[i]] = df_data[HEX_column_list[i]].apply(lambda x: HEX_to_INT(str(x)))
        #     # print(df_data[HEX_column_list[i]])
        # print(" ==================== HEX to INT is END =================== ")

        # # MAC to INT
        # print(" ==================== MAC to INT is START =================== ")
        # MAC_size_columns_num = len(MAC_column_list)
        # print(MAC_size_columns_num)
        # print(MAC_column_list)

        # for i in range(MAC_size_columns_num):
        #     df_data[MAC_column_list[i]] = df_data[MAC_column_list[i]].apply(lambda x: MAC_to_INT(str(x)))
        #     # print(df_data[MAC_column_list[i]])
        # print(" ==================== MAC to INT is END =================== ")


        # IP to INT Modified by : Patrick Yan 2022/12/19 02:49
        print(" ==================== IP to INT is START =================== ")
        IP_size_columns_num = len(IP_column_list)
        print(IP_size_columns_num)
        print(IP_column_list)
        
        for i in range(IP_size_columns_num):
            df_data[IP_column_list[i]] = df_data[IP_column_list[i]].apply(lambda x: IP_to_int(str(x)))
            print(df_data[IP_column_list[i]])
        print(" ==================== IP to INT is END =================== ")
        

        # String to Num Modified by : Patrick Yan 2022/12/19 02:49
        print(" ==================== String to Num is START =================== ")
        feature_dict = {'proto':['tcp','udp','icmp']}
        feature_dict['service'] = ['dns', 'ssl', 'dhcp', 'http', 'ftp', 'dce_rpc', 'gssapi', 'smb;gssapi', 'smb']
        feature_dict['conn_state'] = ['OTH', 'S0', 'SHR', 'RSTRH', 'SH', 'RSTOS0', 'RSTR', 'RSTO', 'SF', 'S1', 'S3', 'S2', 'REJ']
        feature_dict['http_user_agent'] = ['-', 'Microsoft-Windows/10.0 UPnP/1.0', 'DAFUPnP', 'Microsoft-Delivery-Optimization/10.0', 'MICROSOFT_DEVICE_METADATA_RETRIEVAL_CLIENT', 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:66.0) Gecko/20100101 Firefox/66.0', 'Microsoft NCSI', 'Debian APT-HTTP/1.3 (1.7.4)', 'Microsoft-CryptoAPI/10.0', 'Microsoft-WNS/10.0', 'User-Agent: Microsoft-DLNA DLNADOC/1.50', 'Microsoft-Windows/10.0 UPnP/1.0 Microsoft-DLNA DLNADOC/1.50', 'Microsoft BITS/7.8', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML; like Gecko) Chrome/64.0.3282.140 Safari/537.36 Edge/18.17763', 'Windows-Update-Agent/10.0.10011.16384 Client-Protocol/1.91', 'sqlmap/1.2#stable (http://sqlmap.org)', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.11; rv:49.0) Gecko/20100101 Firefox/49.0', 'Mozilla/5.0 (iPhone; U; CPU iOS 2_0 like Mac OS X; en-us)', 'Comos/0.9_(robot@xyleme.com)', 'Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:5.0) Whistler/20110021 myibrow/5.0.0.0', 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:49.0) Gecko/20100101 Firefox/49.0', 'Mozilla/5.0 (SMART-TV; Linux; Tizen 2.3) AppleWebkit/538.1 (KHTML; like Gecko) SamsungBrowser/1.0 Safari/538.1', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/534.30 (KHTML; like Gecko) Version/4.0 Oupeng/10.2.1.86910 Safari/534.30', 'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/4.0; InfoPath.2; SV1; .NET CLR 2.0.50727; WOW64)', 'DonutP; Windows98SE', 'Mozilla/1.22 (compatible; MSIE 10.0; Windows 3.1)', 'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML; like Gecko) Chrome/53.0.2785.143 Safari/537.36', 'DataCha0s/2.0', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_0) AppleWebKit/537.36 (KHTML; like Gecko) Chrome/53.0.2785.143 Safari/537.36', 'Mozilla/5.0 (Windows NT 6.3; rv:36.0) Gecko/20100101 Firefox/36.0', 'Ruby', 'Mozilla/5.0 (Windows NT 6.1; rv:21.0) Gecko/20130401 Firefox/21.0', 'Mozilla/4.08 [en] (WinNT; I ;Nav)', 'Mozilla/5.0 (X11; U; Linux i686; de-DE; rv:1.6) Gecko/20040207 Firefox/0.8', 'hacking', 'Debian APT-HTTP/1.3 (1.6.6)']
        feature_dict['ssl_version'] = ['TLSv12', 'TLSv13', 'TLSv10']
        feature_dict['ssl_cipher'] = [ 'TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256', 'TLS_AES_128_GCM_SHA256', 'TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256', 'TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384', 'TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA']
        feature_dict['ssl_subject'] = ['CN=settings-win.data.microsoft.com;OU=WSE;O=Microsoft;L=Redmond;ST=WA;C=US', 'CN=*.wns.windows.com', 'CN=sls.update.microsoft.com;OU=DSP;O=Microsoft;L=Redmond;ST=WA;C=US', 'CN=*.events.data.microsoft.com;OU=Microsoft;O=Microsoft Corporation;L=Redmond;ST=WA;C=US', 'CN=Mr Shepherd;OU=Security Shepherd Project;O=OWASP;L=Dublin;ST=Ireland;C=IE']
        feature_dict['ssl_issuer'] = [ 'CN=Microsoft Secure Server CA 2011;O=Microsoft Corporation;L=Redmond;ST=Washington;C=US', 'CN=Microsoft IT TLS CA 5;OU=Microsoft IT;O=Microsoft Corporation;L=Redmond;ST=Washington;C=US', 'CN=Microsoft Update Secure Server CA 2.1;O=Microsoft Corporation;L=Redmond;ST=Washington;C=US', 'CN=Mr Shepherd;OU=Security Shepherd Project;O=OWASP;L=Dublin;ST=Ireland;C=IE']  
        feature_dict['http_orig_mime_types'] = [ 'application/xml', 'application/soap+xml']
        feature_dict['http_resp_mime_types'] = [ 'application/ocsp-response', 'application/xml', 'text/json', 'text/plain', 'application/x-debian-package', 'application/vnd.ms-cab-compressed', 'image/png', 'image/jpeg', 'text/html']
        feature_dict['weird_name'] = ['bad_TCP_checksum',  'bad_UDP_checksum', 'active_connection_reuse', 'data_before_established', 'inappropriate_FIN', 'above_hole_data_without_any_acks', 'DNS_RR_unknown_type', 'TCP_ack_underflow_or_misorder', 'dnp3_corrupt_header_checksum', 'possible_split_routing', 'connection_originator_SYN_ack']
        feature_dict['http_method'] = ['GET', 'POST', 'HEAD']
        for feature_array in feature_dict.items() :
            for name in feature_array[1] :
                print(name,'  ',feature_array[1].index(name)+1)
                df_data[feature_array[0]] = df_data[feature_array[0]].replace(name, feature_array[1].index(name)+1)
        print(" ==================== String to Num is START =================== ")

        # type name to Num Modified by : Patrick Yan 2022/12/19 02:49
        df_data['type'] = df_data['type'].replace('normal', 0)
        df_data['type'] = df_data['type'].replace('scanning', 1)
        df_data['type'] = df_data['type'].replace('dos', 3)
        df_data['type'] = df_data['type'].replace('injection', 4)
        df_data['type'] = df_data['type'].replace('ddos', 5)
        df_data['type'] = df_data['type'].replace('password', 6)
        df_data['type'] = df_data['type'].replace('xss', 7)
        df_data['type'] = df_data['type'].replace('ransomware', 8)
        df_data['type'] = df_data['type'].replace('backdoor', 9)
        df_data['type'] = df_data['type'].replace('mitm', 10)
        # print(df_data['type'])

        print("=========== Final Data =============")
        print(df_data)

        # Save to CSV
        df_data.to_csv(path_save + "01_Normalization/" + fileName + "_mapped.csv", index=False)
        print("\tSaved: " + path_save + "01_Normalization/" + fileName + "_mapped.csv")

        # 另存 type 欄位
        type_label = df_data.iloc[:, [44]]
        type_label.to_csv(path_save + "01_Normalization/" + fileName + "_TypeLabel.csv", index = False)

        # 另存 label 欄位
        type_label = df_data.iloc[:, [43]]
        type_label.to_csv(path_save + "01_Normalization/" + fileName + "_Label.csv", index = False)

        # 移除 type 欄位
        df_data = df_data.drop('type', axis=1)
        # print("New columns list length : ", len(df_data))
        print(df_data)
        
        # 移除 label 欄位
        df_data = df_data.drop('label', axis=1)
        # print("New columns list length : ", len(df_data))

        # 移除 dns_query 欄位
        df_data = df_data.drop('dns_query', axis=1)
        # print("New columns list length : ", len(df_data))

        # 移除 http_uri 欄位
        df_data = df_data.drop('http_uri', axis=1)
        # print("New columns list length : ", len(df_data))

        # Save to CSV (Not Type)
        df_data.to_csv(path_save + "01_Normalization/" + fileName + "_mapped_NotType.csv", index=False)
        print("\tSaved: " + path_save + "01_Normalization/" + fileName + "_mapped_NotType.csv")



endTime = datetime.datetime.now()
print('==================== DONE: ', endTime, '====================\n')
print('Duration:', endTime - startTime)


'''
# 算一特徵各欄位(字串)出現幾次
for i, file in enumerate(os.listdir(path_save + '01_Normalization/')):
    if file.endswith(".csv"):
        fileName = file.replace(".csv", '')
        print('\n', counter_File, ':\t', fileName)
        counter_File += 1

        df_data_mapped = pd.read_csv(path_save + '01_Normalization/' + file, sep=',', low_memory=False, index_col=False)
        wlan_mgt_ssid = df_data_mapped['wlan_mgt.ssid']
        print(wlan_mgt_ssid)
        length_wlan_mgt_ssid = len(wlan_mgt_ssid) - 1

        count_wlan_mgt_ssid = pd.Series(list(wlan_mgt_ssid)).value_counts()
        print(count_wlan_mgt_ssid)
        len_wlan_mgt_ssid = len(count_wlan_mgt_ssid)
        print(len_wlan_mgt_ssid)
'''

# ======================================================================================================================
# ======================================================================================================================
# Normalization MinMAX

# PATH
path_r_raw_mapped_dataset = 'Data/01_Normalization/'
path_save = 'Data/01_Normalization/'
path_r_newindex = 'New_Columns.txt'

new_column_list = loadColumns(path_r_newindex) # Not sidd and type


print("========================================  Big Data is Start MinMax ======================================")

# Create Folder
createFolder(path_save + "BigData_MinMax")

startTime = datetime.datetime.now()
print('==================== Start Find Min and Max : ', startTime, '====================')

BigData_max_list = []
BigData_min_list = []
counter_File = 0

for i, file in enumerate(os.listdir(path_r_raw_mapped_dataset)):
    if file.endswith("_Big_mapped_NotType.csv"):
        fileName = file.replace(".csv", '')
        print('\n', counter_File, ':\t', fileName)
        counter_File += 1

        # load raw_dataset
        df_data_mapped = pd.read_csv(path_r_raw_mapped_dataset + file, sep=',', low_memory=False, index_col=False)
        print(df_data_mapped)

        ''' Check MAX MIN '''
        # find max and min for current file
        BigData_max_df = df_data_mapped.max()
        BigData_min_df = df_data_mapped.min()

        # put current max and min into list
        BigData_max_list.append(BigData_max_df.tolist())
        BigData_min_list.append(BigData_min_df.tolist())
        print(BigData_max_list)
        print(BigData_min_list)

# save all max and min value into a dataframe
BigData_max_df = pd.DataFrame(np.array(BigData_max_list), dtype='float64')
BigData_min_df = pd.DataFrame(np.array(BigData_min_list), dtype='float64')

BigData_max_df.to_csv(path_save + 'BigData_MinMax/' + "BigData_max_each.csv", index=None, header=None)
BigData_min_df.to_csv(path_save + 'BigData_MinMax/' + "BigData_min_each.csv", index=None, header=None)

# find the max value in all max dataframe
BigData_max_min_list = []
BigData_max_min_list.append(BigData_max_df.max())
BigData_max_min_list.append(BigData_min_df.min())

# save max_min_columns names
with open(path_save + 'BigData_MinMax/' + "BigData_columns_max_min.txt", 'w') as f:
    for column in df_data.columns:
        f.write("%s\n" % column)

BigData_column_max_min_list = loadColumns(path_save + 'BigData_MinMax/' + "BigData_columns_max_min.txt")
BigData_max_min = pd.DataFrame(np.array(BigData_max_min_list), dtype='float64')
BigData_max_min.to_csv(path_save + 'BigData_MinMax/' + "BigData_max_min.csv", index=False, header=None)

endTime = datetime.datetime.now()
print('==================== DONE Find Min and Max : ', endTime, '====================\n')
print('Duration:', endTime - startTime)

# --------------------------------------------------------------------------------------------------------------

startTime = datetime.datetime.now()
print('==================== Start Run MinMax : ', startTime, '====================')

# For minmax
df_BigData_max_min = pd.read_csv(path_save + 'BigData_MinMax/' + 'BigData_max_min.csv', sep=',', header=None, names=new_column_list)

counter_File = 0
for i, file in enumerate(os.listdir(path_r_raw_mapped_dataset)):
    if file.endswith('_Big_mapped_NotType.csv'):
        fileName = file.replace('_mapped_NotType.csv', '')
        print('\n', counter_File, ':\t', fileName)
        counter_File += 1

        # Load raw_dataset
        df_data_BigData_mapped = pd.read_csv(path_r_raw_mapped_dataset + file, header=0, low_memory=False)

        print('\t\tNormalizing....')

        # MinMax
        for j, column in enumerate(list(new_column_list)):
            max = df_BigData_max_min[column].max()
            min = df_BigData_max_min[column].min()
            df_data_BigData_mapped[column] = df_data_BigData_mapped[column].apply(lambda x: minmax(float(x), max, min)).astype(np.float64)
            print('\t' + str(j) + '\t' + column)

        # save as csv file
        df_data_BigData_mapped.to_csv(path_save + fileName + "_Normalization.csv", index=False)
        print("\tSaved")

endTime = datetime.datetime.now()
print('==================== DONE Run MinMax : ', endTime, '====================\n')
print('Duration:', endTime - startTime)

print("========================================  Big Data is End MinMax  ======================================")

# ----------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------

print("========================================  Small Data is Start MinMax ======================================")

# Create Folder
createFolder(path_save + "SmallData_MinMax")

startTime = datetime.datetime.now()
print('==================== Start Find Min and Max : ', startTime, '====================')

SmallData_max_list = []
SmallData_min_list = []
counter_File = 0

for i, file in enumerate(os.listdir(path_r_raw_mapped_dataset)):
    if file.endswith("_Small_mapped_NotType.csv"):
        fileName = file.replace(".csv", '')
        print('\n', counter_File, ':\t', fileName)
        counter_File += 1

        # load raw_dataset
        df_data_mapped = pd.read_csv(path_r_raw_mapped_dataset + file, sep=',', low_memory=False, index_col=False)
        print(df_data_mapped)

        ''' Check MAX MIN '''
        # find max and min for current file
        SmallData_max_df = df_data_mapped.max()
        SmallData_min_df = df_data_mapped.min()

        # put current max and min into list
        SmallData_max_list.append(SmallData_max_df.tolist())
        SmallData_min_list.append(SmallData_min_df.tolist())
        print(SmallData_max_list)
        print(SmallData_min_list)

# save all max and min value into a dataframe
SmallData_max_df = pd.DataFrame(np.array(SmallData_max_list), dtype='float64')
SmallData_min_df = pd.DataFrame(np.array(SmallData_min_list), dtype='float64')

SmallData_max_df.to_csv(path_save + 'SmallData_MinMax/' + "SmallData_max_each.csv", index=None, header=None)
SmallData_min_df.to_csv(path_save + 'SmallData_MinMax/' + "SmallData_min_each.csv", index=None, header=None)

# find the max value in all max dataframe
SmallData_max_min_list = []
SmallData_max_min_list.append(SmallData_max_df.max())
SmallData_max_min_list.append(SmallData_min_df.min())

# save max_min_columns names
with open(path_save + 'SmallData_MinMax/' + "SmallData_columns_max_min.txt", 'w') as f:
    for column in df_data.columns:
        f.write("%s\n" % column)

SmallData_column_max_min_list = loadColumns(path_save + 'SmallData_MinMax/' + "SmallData_columns_max_min.txt")
SmallData_max_min = pd.DataFrame(np.array(SmallData_max_min_list), dtype='float64')
SmallData_max_min.to_csv(path_save + 'SmallData_MinMax/' + "SmallData_max_min.csv", index=False, header=None)

endTime = datetime.datetime.now()
print('==================== DONE Find Min and Max : ', endTime, '====================\n')
print('Duration:', endTime - startTime)

# --------------------------------------------------------------------------------------------------------------

startTime = datetime.datetime.now()
print('==================== Start Run MinMax : ', startTime, '====================')

# For minmax
df_SmallData_max_min = pd.read_csv(path_save + 'SmallData_MinMax/' + 'SmallData_max_min.csv', sep=',', header=None, names=new_column_list)

counter_File = 0
for i, file in enumerate(os.listdir(path_r_raw_mapped_dataset)):
    if file.endswith('_Small_mapped_NotType.csv'):
        fileName = file.replace('_mapped_NotType.csv', '')
        print('\n', counter_File, ':\t', fileName)
        counter_File += 1

        # Load raw_dataset
        df_data_SmallData_mapped = pd.read_csv(path_r_raw_mapped_dataset + file, header=0, low_memory=False)

        print('\t\tNormalizing....')

        # MinMax
        for j, column in enumerate(list(new_column_list)):
            max = df_SmallData_max_min[column].max()
            min = df_SmallData_max_min[column].min()
            df_data_SmallData_mapped[column] = df_data_SmallData_mapped[column].apply(lambda x: minmax(float(x), max, min)).astype(np.float64)
            print('\t' + str(j) + '\t' + column)

        # save as csv file
        df_data_SmallData_mapped.to_csv(path_save + fileName + "_Normalization.csv", index=False)
        print("\tSaved")

endTime = datetime.datetime.now()
print('==================== DONE Run MinMax : ', endTime, '====================\n')
print('Duration:', endTime - startTime)

print("========================================  Small Data is End MinMax ======================================")