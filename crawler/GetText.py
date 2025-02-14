import requests
import csv

headers = {
    'cookie' : 'SINAGLOBAL=8113781381633.099.1733739068339; XSRF-TOKEN=yIp23fYe2__NpaYRhe3nzVih; _s_tentry=passport.weibo.com; appkey=; Apache=1541707329655.7483.1735978503535; ULV=1735978503536:2:1:1:1541707329655.7483.1735978503535:1733739068342; SCF=ArBPmutwssd5labr5TdVcN9IOO7PLq_ivMStbWlwh_GUqdkqe1BvSzjSYmS0mW2l-6N2L932im_hUzbpE1xjVwI.; SUB=_2A25Ko4eADeRhGeFH7FAX9i7KyjqIHXVpwIVIrDV8PUNbmtANLWjhkW9NepOKdgLabXWpes1wBmwBL2iz8FeVUFxu; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WFlvkPMBllyNgzm0HVLzpJk5NHD95QN1KMESoq7So2cWs4DqcjMi--NiK.Xi-2Ri--ciKnRi-zNSK.N1h-fe0MRSntt; ALF=02_1741653200; WBPSESS=dfxMABdxgRAFxpcS5UT-DFOpu5t7HHfmwgdyOF5QmQRW-E5z9LVqlg8Vwo1XpvtKQ7IN_57qBCv0PSSgI2stwfDz25jCldYWTu2Hq4rGg1Yde97Vs4YH5swrWcDQ1YbmPhBEFKRpligkkfUCMeeCfQ==',

    'referer' : 'https://weibo.com/3611115253/Pcgnj4yCT',

    'user-agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36'
}

f = open('review.csv', mode='a', encoding='utf-8-sig', newline='')
csv_write = csv.writer((f))
csv_write.writerow(['text_raw','created_at'])



def get_next(next='count=10'):
    url = f'https://weibo.com/ajax/statuses/buildComments?is_reload=1&id=5003170104741808&is_show_bulletin=2&is_mix=0&{next}&uid=7190522839&fetch_level=0&locale=zh-CN'

    response = requests.get(url=url, headers=headers)
    json_data = response.json()


    if 'data' in json_data :
        data_list = json_data['data']
        max_id = json_data['max_id']
        for data in data_list:
            text_raw = data['text_raw']
            created_at = data['created_at']
            csv_write.writerow([text_raw, created_at])
        max_str = 'max_id=' + str(max_id)
        get_next(max_str)
    else:
        print("返回的数据中缺少 'data'", json_data)
        f.close()

get_next()





