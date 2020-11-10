"""
기간: 2020년 8월~ 2020년 11월
프로젝트명: 전기자동차 충전 데이터를 이용한 충전기 예측 추천 모델 개발
프로젝트 내용
1. 전기자동차 충전기 이용 데이터를 활용하여 사용자의 이용 패턴을 분석하여
학습된 모델을 이용하여 전기차 사용자가 원하는 시간과 장소에 최적의 충전소를 표시해준다.
2. Flask를 활용하여 웹페이지를 만들어 사용자들이 편리하게 서비스를 활용할 수 있도록 해준다.
"""



from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os, re, joblib
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow import keras
from keras.models import load_model
import requests
import pprint
import folium
import branca
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from folium.features import DivIcon
import folium.plugins as plugins

app = Flask(__name__) # flask name 선언
app.debug = True

model_ev_dt_a = None
model_ev_dt_b = None
model_ev_dt_c = None
model_ev_dt_d = None
model_ev_dt_e = None
model_ev_dt_f = None

def load_ev():
    global model_ev_dt_a, model_ev_dt_b, model_ev_dt_c, model_ev_dt_d, model_ev_dt_e, model_ev_dt_f
    model_ev_dt_a = joblib.load(os.path.join(app.root_path, 'static/model/ev_a.pkl'))
    model_ev_dt_b = joblib.load(os.path.join(app.root_path, 'static/model/ev_b.pkl'))
    model_ev_dt_c = joblib.load(os.path.join(app.root_path, 'static/model/ev_c.pkl'))
    model_ev_dt_d = joblib.load(os.path.join(app.root_path, 'static/model/ev_d.pkl'))
    model_ev_dt_e = joblib.load(os.path.join(app.root_path, 'static/model/ev_e.pkl'))
    model_ev_dt_f = joblib.load(os.path.join(app.root_path, 'static/model/ev_f.pkl'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/save_map', methods=['GET', 'POST'])
def save_map():

    A = pd.read_csv('D:\workspace\web\project\ev_project_web\static\data\ev\A.csv')
    B = pd.read_csv('D:\workspace\web\project\ev_project_web\static\data\ev\B.csv')
    C = pd.read_csv('D:\workspace\web\project\ev_project_web\static\data\ev\C.csv')
    D = pd.read_csv('D:\workspace\web\project\ev_project_web\static\data\ev\D.csv')
    E = pd.read_csv('D:\workspace\web\project\ev_project_web\static\data\ev\E.csv')
    F = pd.read_csv('D:\workspace\web\project\ev_project_web\static\data\ev\F.csv')

    a_place = ['한경면', '한림읍', '애월읍']
    b_place = ['추자면', '연동', '이도2동', '오라2동', '해안동', '아라1동',
                          '건입동', '노형동', '영평동', '삼양2동', '오라1동', '도남동',
                          '일도2동', '용강동', '외도1동', '봉개동', '아라2동', '아라동',
                          '도두1동', '이도1동', '화북1동', '용담2동', '오등동', '도련2동',
                          '용담1동', '삼도2도']
    c_place = ['조천읍', '구좌읍']
    d_place = ['대정읍', '안덕면', '하예동', '중문동', '대천동', '상예동',
                          '색달동']
    e_place = ['하효동', '남원읍', '강정동', '서홍동', '법환동', '동홍동',
                          '토평동', '서귀동', '보목동', '서호동', '회수동', '상효동']
    f_place = ['성산읍', '표선면']
    date = int(request.form['month'] + request.form['date']) 
    day = int(request.form['day'])
    times = int(request.form['times'])
    city = request.form['city']
    gu = request.form['gu']
    
    
    for i in a_place:
        if i == gu:
            test = A.drop_duplicates('cid')
            test['week'] = date
            test['day'] = day
            test['time'] = times
            aa = test.drop(['use', 'sid', 'gu'], axis = 1)
            pred = model_ev_dt_a.predict(aa)
            apred = pd.DataFrame({'dt': pred})
            test['DT'] = list(apred['dt'])
            test = test.sort_values(by=['gu', 'sid'], ascending=True)
            test = test.drop(['Unnamed: 0', 'cid', 'week', 'use'], axis=1)
            break

    for i in b_place:
        if i == gu:
            test = B.drop_duplicates('cid')
            test['week'] = date
            test['day'] = day
            test['time'] = times
            bb = test.drop(['use', 'sid', 'gu'], axis = 1)
            pred = model_ev_dt_b.predict(bb)
            bpred = pd.DataFrame({'dt': pred})
            test['DT'] = list(bpred['dt'])
            test = test.sort_values(by=['gu', 'sid'], ascending=True)
            test = test.drop(['Unnamed: 0', 'cid', 'week', 'use'], axis=1)
            break
    for i in c_place:
        if i == gu:
            test = C.drop_duplicates('cid')
            test['week'] = date
            test['day'] = day
            test['time'] = times
            cc = test.drop(['use', 'sid', 'gu'], axis = 1)
            pred = model_ev_dt_c.predict(cc)
            cpred = pd.DataFrame({'dt': pred})
            test['DT'] = list(cpred['dt'])
            test = test.sort_values(by=['gu', 'sid'], ascending=True)
            test = test.drop(['Unnamed: 0', 'cid', 'week', 'use'], axis=1)
            break
    for i in d_place:
        if i == gu:
            test = D.drop_duplicates('cid')
            test['week'] = date
            test['day'] = day
            test['time'] = times
            dd = test.drop(['use', 'sid', 'gu'], axis = 1)
            pred = model_ev_dt_d.predict(dd)
            dpred = pd.DataFrame({'dt': pred})
            test['DT'] = list(dpred['dt'])
            test = test.sort_values(by=['gu', 'sid'], ascending=True)
            test = test.drop(['Unnamed: 0', 'cid', 'week', 'use'], axis=1)
            break
    for i in e_place:
        if i == gu:
            test = E.drop_duplicates('cid')
            test['week'] = date
            test['day'] = day
            test['time'] = times
            ee = test.drop(['use', 'sid', 'gu'], axis = 1)
            pred = model_ev_dt_e.predict(ee)
            epred = pd.DataFrame({'dt': pred})
            test['DT'] = list(epred['dt'])
            test = test.sort_values(by=['gu', 'sid'], ascending=True)
            test = test.drop(['Unnamed: 0', 'cid', 'week', 'use'], axis=1)
    for i in f_place:
        if i == gu:
            test = F.drop_duplicates('cid')
            test['week'] = date
            test['day'] = day
            test['time'] = times
            ff = test.drop(['use', 'sid', 'gu'], axis = 1)
            pred = model_ev_dt_f.predict(ff)
            fpred = pd.DataFrame({'dt': pred})
            test['DT'] = list(fpred['dt'])
            test = test.sort_values(by=['gu', 'sid'], ascending=True)
            test = test.drop(['Unnamed: 0', 'cid', 'week', 'use'], axis=1)
            break
    top5 = []
    t_len = len(test)
    test.reset_index(inplace=True)
    for i in range(t_len): 
        if test['gu'][i] == gu :
            if i == 0 :
                top5.append(test['sid'][i])
            elif test['gu'][i-1] != test['gu'][i]:
                top5.append(test['sid'][i])

    if len(top5) <=5:
        for k in range(t_len):
            if test['gu'][k]!=gu :
                if k == 0:
                    top5.append(test['sid'][k])
                elif test['gu'][k-1] != test['gu'][k]:
                    top5.append(test['sid'][k])
    top5 = top5[0:5]

    df = pd.read_csv('D:\workspace\web\project\ev_project_web\static\data\충전소추천c_add\제주도_통계_수정.csv', sep=',', encoding = 'euc-kr', usecols = ['sid','week','time','use'])
    for i in range(0,len(df)):
        if df['use'][i] > 2 :
            df['use'][i] = 2
    df['카운트'] = 1
    df = df.groupby(by=['sid','week','time']).sum()
    df['확률'] = round((df['use']/df['카운트']/2),2)

    a = top5
    day_check = test['day']

    df_add = pd.read_csv('D:\workspace\web\project\ev_project_web\static\data\충전소추천c_add\c_name.csv', sep=',',usecols = ['c_name','sid','c_add'])
    df_add=pd.DataFrame(df_add)

    name = []
    sid = []
    add = []

    for i in range(len(df_add)):
        if i == 0 :
            name.append(df_add['c_name'][i])
            sid.append(df_add['sid'][i])
            add.append(df_add['c_add'][i])
        elif df_add['sid'][i-1] != df_add['sid'][i]:
            name.append(df_add['c_name'][i])
            sid.append(df_add['sid'][i])
            add.append(df_add['c_add'][i])
    charger_dict = {"name": name, "sid": sid, "c_add": add}
    charger_recommend = pd.DataFrame(charger_dict)
    dic = {"sid": a}
    input_charger = pd.DataFrame(dic)
    charger_chk = pd.merge(input_charger,charger_recommend,on='sid')

    charger_chk['위도'] = 0.000
    charger_chk['경도'] = 0.000

    b=[]
    for c in range(len(charger_chk)):
        b.append(charger_chk['name'][c])
    day = ['월요일','화요일','수요일','목요일','금요일','토요일','일요일']

    for i in range(len(df)):
        for j in range(len(a)):
            df_cid=df.loc[a[j]]
            for k in range(7):
                plt.cla()
                df_week=df_cid.loc[k]
                df_time=df_week
                df_time


                plt.rcParams['figure.figsize'] = [6,3.4]
                plt.rcParams["font.family"] = 'Malgun Gothic'

                plt.plot(df_time.index, (df_time.확률*100), marker='s', color='green', markersize = 5)

                plt.rc('xtick', labelsize = 10)
                plt.rc('ytick', labelsize = 10)
                plt.xticks(np.arange(0, 24, 1), labels=['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10','11','12','13','14','15','16','17','18','19','20','21','22','23','24'])
                plt.yticks(np.arange(0, 100,10), ('10', '20', '30', '40', '50', '60', '70', '80', '90', '100'))
                plt.title('%s - 이용자 통계(%s)'%(b[j],day[k]), fontsize=15)
                plt.ylabel('PERCENT(%)', fontsize=10)
                plt.xlabel('TIME', fontsize=10)
                plt.grid(True)
                plt.savefig('static/image/fig%.d-%.d.png'%(j,k))
                
                
                
        break

    for i in range(len(charger_chk)):
        location = charger_chk['c_add'][i]

        URL = 'https://maps.googleapis.com/maps/api/geocode/json?key=AIzaSyCNgb2Qc2tOS8zo4xr7odc9WigwW9h6noM' \
        '&sensor=false&language=ko&address={}'.format(location)

        response = requests.get(URL)
        data = response.json()

        lat = data['results'][0]['geometry']['location']['lat']
        lng = data['results'][0]['geometry']['location']['lng']
        
        u_address = []
        u = data['results'][0]
        u_address=u['formatted_address']
        
        lat =[]
        lng =[]
        k=data['results'][0]
        k_1=k['geometry']['location']
        k_1
        lat = k_1['lat'] #위도
        lng = k_1['lng'] #경도
        charger_chk['위도'][i] = lat
        charger_chk['경도'][i] = lng

        a_위도 = 0.00
    a_경도 = 0.00
    #전체 줌 위치
    for k in range(len(charger_chk)):
        a_위도 += charger_chk['위도'][k]
        a_경도 += charger_chk['경도'][k]

    a_위도 = a_위도/len(charger_chk)
    a_경도 = a_경도/len(charger_chk)

    imgs = '<img src="static/image/box.png">'

    m = folium.Map(
        location=[a_위도,a_경도],
        zoom_start=13,

    )
    
    All = folium.plugins.MarkerCluster(control=False, name='Top5')
    m.add_child(All)
    


    #실제 주소 받고 마커 체크
    for i in range(len(charger_chk)):
        pic_1 = base64.b64encode(open('static/image/%s.png'%(charger_chk['sid'][i]),'rb').read()).decode()
        pic_2 = base64.b64encode(open('static/image/fig%d-%d.png'%(i, day_check[i]),'rb').read()).decode()
        image_tag = '''<img src="data:image/jpeg;base64,{0}"><br><img src="data:image/jpeg;base64,{1}" style="width:430px; height:230px"><br>'''.format(pic_1, pic_2)
        iframe = folium.IFrame(image_tag, width=450, height=520)
        popup = folium.Popup(iframe, max_width=600)
        if i == 0:
            name2 = folium.plugins.FeatureGroupSubGroup(All, '<div><div style="float: left; "><img src="static/image/list_icon_green.ico" style ="width:40px; height:40px;">  </div><h4 style="font-weight: bold; font-family:Lato">%s</h4><h5 style=" font-family:Lato">%s</h5></div>'%(charger_chk['name'][i],charger_chk['c_add'][i]))
            m.add_child(name2)
            
            icon=folium.Icon(color='green', icon='car', icon_color="white", prefix='fa')
            folium.Marker(
            location=[charger_chk['위도'][i],charger_chk['경도'][i]],
            popup=popup,tooltip='%s'%(charger_chk['name'][i]),
            icon=icon
            ).add_to(m)
            
            folium.Marker(location=[charger_chk['위도'][i],charger_chk['경도'][i]], icon=DivIcon(
        icon_size=(120,24),
        icon_anchor=(3,5),
        
        html='<div style="position:relative;text-align:center; width:200px;"><img src="static/image/box.png"><div style="position: absolute; bottom: 18px; left: 10px; text-align:center; font-size:12pt; font-weight:bold; font-family:Lato">%s</div></div>'%(charger_chk['name'][i]),
        )).add_to(m)
            

        else :
            name2 = folium.plugins.FeatureGroupSubGroup(All, '<div><div style="float: left;"><img src="static/image/list_icon_blue.ico" style ="width:40px; height:40px;"></div>  <h4 style="font-weight: bold; font-family:Lato">%s</h4><h5 style=" font-family:Lato">%s</h5></div>'%(charger_chk['name'][i],charger_chk['c_add'][i]))
            m.add_child(name2)
            icon=folium.Icon(color='blue', icon='car', icon_color="white", prefix='fa')
            folium.Marker(
            location=[charger_chk['위도'][i],charger_chk['경도'][i]],
            popup=popup,tooltip='%s'%(charger_chk['name'][i]),
            icon=icon
            ).add_to(m)

            folium.Marker(location=[charger_chk['위도'][i],charger_chk['경도'][i]], icon=DivIcon(
        icon_size=(120,24),
        icon_anchor=(3,5),
        html='<div style="position:relative;text-align:center; width:200px;"><img src="static/image/box.png"><div style="position: absolute; bottom: 18px; left: 10px; text-align:center; font-size:12pt; font-weight: bold; font-family:Lato">%s</div></div>'%(charger_chk['name'][i]),
        )).add_to(m)
            

    folium.LayerControl(position='topleft', collapsed=False).add_to(m)       
    #m._repr_html_()
    m.save('templates/map.html')
    return render_template('map.html')

        
if __name__ == '__main__':
    load_ev()
    app.debug = True
    app.run(host='0.0.0.0') # host주소와 port number 선언