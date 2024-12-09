import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import plotly.express as px
import requests
import xml.etree.ElementTree as ET
from datetime import timedelta, date

st.set_page_config(layout="wide")


def get_data(cod_station, start_date, end_date):
    url = "https://telemetriaws1.ana.gov.br/ServiceANA.asmx"
    headers = {"Content-Type": "application/soap+xml; charset=utf-8"}
    soap_body = """<?xml version="1.0" encoding="utf-8"?>
    <soap12:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
                     xmlns:xsd="http://www.w3.org/2001/XMLSchema" 
                     xmlns:soap12="http://www.w3.org/2003/05/soap-envelope">
      <soap12:Body>
        <DadosHidrometeorologicos xmlns="http://MRCS/">
          <codEstacao>{codEstacao}</codEstacao>
          <dataInicio>{dataInicio}</dataInicio>
          <dataFim>{dataFim}</dataFim>
        </DadosHidrometeorologicos>
      </soap12:Body>
    </soap12:Envelope>"""
    payload = soap_body.format(codEstacao=cod_station, dataInicio=start_date, dataFim=end_date)
    response = requests.post(url, headers=headers, data=payload)
    return response.text


def parse_xml(soap_response):
    root = ET.fromstring(soap_response)
    ns = {'soap': 'http://www.w3.org/2003/05/soap-envelope'}
    data_root = root.find('.//DocumentElement', ns)
    records = []
    for item in data_root.findall('./DadosHidrometereologicos'):
        record = {
            "cod_station": item.findtext('CodEstacao'),
            "date_time": item.findtext('DataHora'),
            "level": item.findtext('Nivel'),
            "rainfall": item.findtext('Chuva'),
        }
        records.append(record)
    return pd.DataFrame(records)


@st.cache_data
def load_data(end_date):
    soap_response = get_data(
        cod_station = "86510000", # MUÇUM station -- Lat -29.16720 Long -51.86860
        start_date = "01/01/2019",
        end_date = end_date
    )
    df = parse_xml(soap_response)
    df['day'] = pd.to_datetime(df.date_time).dt.date
    df['level'] = pd.to_numeric(df['level'])
    df['rainfall'] = pd.to_numeric(df['rainfall'])
    df_agg = df.groupby('day').agg({'level': 'max', 'rainfall': 'sum'})
    return df_agg.reset_index()


def detect_anomalies(df):
    data_rainfall = df.rename(columns={"day": "ds", "rainfall": "y"})
    model_rainfall = Prophet(seasonality_mode='multiplicative')
    model_rainfall.add_seasonality(name="annual", period=365.25, fourier_order=8)
    model_rainfall.fit(data_rainfall)
    future = model_rainfall.make_future_dataframe(periods=FUTURE_DAYS, include_history=False)
    forecast_rainfall = model_rainfall.predict(future).rename(columns={"yhat": "rainfall"})
    model = Prophet(seasonality_mode='multiplicative')
    model.add_seasonality(name="annual", period=365.25, fourier_order=8)
    model.add_regressor("rainfall")
    data = df.rename(columns={"day": "ds", "level": "y"})
    model.fit(data)
    forecast = model.predict(pd.concat([data[['ds','rainfall']], forecast_rainfall[['ds','rainfall']]]))
    forecast["y"] = data["y"]
    forecast["anomaly"] = forecast['y'] > forecast["yhat_upper"]
    return forecast


def plot_anomalies(data):
    data = data.query(f"ds >= '{START_FILTER.strftime('%Y-%m-%d')}' ")
    fig = px.line(data, x="ds", y=["y", "yhat"], labels={"ds": "date"})
    fig.add_scatter(x=data.loc[data["anomaly"], "ds"], y=data.loc[data["anomaly"], "y"], mode='markers', name="Anomaly", marker=dict(color='red', size=7))        
    fig.update_traces(line=dict(color='grey', dash='dot'), selector=dict(name="yhat"))
    future_start = date.today().strftime("%Y-%m-%d")
    future_end = (date.today() + timedelta(days=FUTURE_DAYS)).strftime("%Y-%m-%d")
    fig.add_vrect(x0=future_start, x1=future_end,fillcolor="LightBlue", opacity=0.2,layer="below", line_width=0)
    fig.add_annotation(x=future_start, y=900, text="Forecasting", showarrow=True, arrowhead=2,arrowsize=1,arrowwidth=2,arrowcolor="grey")
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)






today = date.today().strftime("%d/%m/%Y")
data = load_data(today)

col1, _ = st.columns([20, 1])


with col1:
    
    col1_1, col1_2, _ = st.columns([6, 4, 10])
    
    with col1_1:

        with st.container(border=True):
            st.write("###### Plot start date")
            START_FILTER = st.slider("Plot start date", min_value=date(2019, 1, 1), max_value=date.today() - timedelta(days=1), value=date(2022, 1, 1), label_visibility="collapsed", format="DD/MM/YY")
                
    
    with col1_2:

        with st.container(border=True):
            st.write("###### Forecasting (days)")
            FUTURE_DAYS = st.slider('Forecasting (days)', 1, 365, 90, label_visibility="collapsed")    
    
            
    st.write(
            """
            # Taquari river level over time
            ##### Measured since 2019 in Muçum, Rio Grande do Sul, Brazil
            """
    )

    plot_anomalies(detect_anomalies(data))

    

    





