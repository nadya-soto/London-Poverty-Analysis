import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from streamlit_folium import folium_static
import folium
from folium.plugins import MarkerCluster
from datetime import datetime

# Configuración de página
st.set_page_config(
    layout="wide",
    page_title="London Poverty Analysis | Data Science Portfolio",
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .st-bw {background-color: white !important;}
    .header-text {font-size:24px !important; color: #2c3e50;}
    .metric-card {border-radius: 10px; padding: 15px; background: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    .highlight {background-color: #e3f2fd; border-left: 4px solid #2196f3; padding: 10px;}
    .footer {text-align: center; padding: 10px; margin-top: 30px; color: #7f8c8d;}
    </style>
    """, unsafe_allow_html=True)

# --- Title and Introduction ---
st.title(" London Poverty Analysis Dashboard")
st.markdown("""
    <div class="highlight">
    <b>Professional Data Science Project</b> - This analysis examines poverty trends across London boroughs from 2018-2023, 
    exploring correlations with unemployment, housing affordability, and economic indicators. The project demonstrates 
    geospatial analysis, time series forecasting, and statistical modeling capabilities.
    </div>
    """, unsafe_allow_html=True)

# --- Función para validar y transformar CRS ---
def validate_and_set_crs(gdf):
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=27700)  # Origen tus datos
    gdf = gdf.to_crs(epsg=4326)       # Para visualización en lat/lon
    gdf["valid"] = gdf["geometry"].is_valid
    return gdf

# --- Cargar datos con mejor manejo de errores ---
@st.cache_data
def load_data():
    try:
        df_la = pd.read_excel("data/LondonPovertyData.xlsx", sheet_name="Local Authority", header=2)
        gdf = gpd.read_file("data/London_Ward_CityMerged.shp")
        gdf.rename(columns={"LAD23NM": "Borough"}, inplace=True)
        gdf = validate_and_set_crs(gdf)
        
        # Datos adicionales para análisis avanzado
        data_pobreza = {
            "Borough": ["Brent", "Camden", "Kensington and Chelsea", "Lambeth", "Wandsworth", "Westminster"],
            2018: [13006, 7108, 2142, 10633, 6976, 4388],
            2019: [14028, 7383, 2181, 11287, 7643, 4464],
            2020: [13306, 6549, 1975, 9994, 7087, 4057],
            2021: [11905, 6423, 2009, 8801, 6503, 3809],
            2022: [11312, 6085, 1954, 8526, 6762, 3522]
        }
        df_pobreza = pd.DataFrame(data_pobreza).set_index("Borough")
        
        data_desempleo = {
            'Borough': ['Brent', 'Camden', 'Kensington and Chelsea', 'Lambeth', 'Wandsworth', 'Westminster'],
            2018: [7900, 4900, 5800, 14200, 5500, 5100],
            2019: [6800, 7400, 3400, 12300, 8000, 4700],
            2020: [3000, 2500, 3900, 16900, 2400, 9800],
            2021: [11300, 0, 1900, 3600, 10800, 3400],
            2022: [4900, 4700, 0, 0, 1900, 2000]
        }
        df_desempleo = pd.DataFrame(data_desempleo).set_index("Borough")
        
        data_precio_casa = {
            "Borough": ["Brent", "Camden", "Kensington and Chelsea", "Lambeth", "Wandsworth", "Westminster"],
            2018: [16.32, 18.23, 33.43, 13.82, 14.80, 23.71],
            2019: [15.59, 18.27, 27.64, 14.41, 14.38, 21.32],
            2020: [14.34, 19.12, 26.23, 13.80, 14.43, 22.45],
            2021: [15.29, 19.40, 24.77, 13.84, 14.68, 21.52],
            2022: [14.77, 19.68, 34.81, 13.17, 14.48, 24.09]
        }
        df_precio_casa = pd.DataFrame(data_precio_casa).set_index("Borough")
        
        cpi_data = {
            'Year': ['2018/19', '2019/20', '2020/21', '2021/22', '2022/23'],
            'CPIH Annual Rate': [2.3, 1.7, 1.0, 2.5, 7.9]
        }
        df_cpi = pd.DataFrame(cpi_data)
        
        return df_la, gdf, df_pobreza, df_desempleo, df_precio_casa, df_cpi
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None, None, None

df_la, gdf, df_pobreza, df_desempleo, df_precio_casa, df_cpi = load_data()

if df_la is None:
    st.stop()

years = ["2018/19", "2019/20", "2020/21", "2021/22", "2022/23"]

# --- Sidebar with improved layout ---
st.sidebar.title("Dashboard Controls")
st.sidebar.markdown("Customize the analysis and visualizations:")

# Borough selection with search functionality
boroughs = df_la["LA"].unique()
selected_borough = st.sidebar.selectbox("Select Borough", boroughs, index=boroughs.tolist().index('Camden'))

# Year range selector
year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=2018,
    max_value=2022,
    value=(2018, 2022)
)

# Visualization options
map_style = st.sidebar.selectbox(
    "Map Style",
    ["OpenStreetMap", "CartoDB Positron"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Project Details:**
- **Data Sources:** London Datastore, ONS
- **Tools:** Python, GeoPandas, Streamlit
- **Author:** Jocelyn Soto
""")

# --- Main Content Tabs ---
tab1, tab2, tab3 = st.tabs(["Overview & Trends", "Geospatial Analysis", "Advanced Analytics"])

with tab1:
    st.header("London Poverty Overview")
    
    # Key Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        current_poverty = df_la[df_la["LA"] == selected_borough][years[-1]].values[0]
        st.metric(f"Current Poverty Cases ({years[-1]})", f"{current_poverty:,.0f}")
    
    with col2:
        change_pct = ((df_la[df_la["LA"] == selected_borough][years[-1]].values[0] - 
                     df_la[df_la["LA"] == selected_borough][years[0]].values[0]) / df_la[df_la["LA"] == selected_borough][years[0]].values[0] * 100)
        st.metric("5-Year Change", f"{change_pct:.1f}%", 
                 "Improvement" if change_pct < 0 else "Increase")
    
    with col3:
        london_avg = df_la[df_la["LA"] != "City of London"][years[-1]].mean()
        diff_from_avg = current_poverty - london_avg
        st.metric("Vs London Average", f"{diff_from_avg:,.0f}", 
                 "Below Average" if diff_from_avg < 0 else "Above Average")
    
    # Borough comparison chart
    st.subheader(f"Poverty Trends: {selected_borough} vs London Average")
    
    borough_data = df_la[df_la["LA"] == selected_borough].iloc[:, 1:]
    borough_data = borough_data.T
    borough_data.columns = ["Poverty Cases"]
    borough_data.index.name = "Year"
    
    london_avg = df_la[df_la["LA"] != "City of London"].iloc[:, 1:].mean()
    comparison_df = pd.DataFrame({
        "Year": borough_data.index,
        selected_borough: borough_data["Poverty Cases"].values,
        "London Average": london_avg.values
    }).set_index("Year")
    
    fig = px.line(comparison_df, 
                 x=comparison_df.index, 
                 y=[selected_borough, "London Average"],
                 markers=True,
                 title=f"Poverty Trend Comparison (2018-2023)",
                 labels={"value": "Poverty Cases", "variable": "Area"},
                 color_discrete_sequence=["#3498db", "#e74c3c"])
    
    fig.update_layout(
        hovermode="x unified",
        xaxis_title="Year",
        yaxis_title="Poverty Cases",
        legend_title="Area"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Indicator Correlations")
    
    # Prepare correlation data
    corr_data = pd.concat([
        df_pobreza.stack().rename('Child Poverty'),
        df_desempleo.stack().rename('Unemployment'),
        df_precio_casa.stack().rename('House Price Ratio')
    ], axis=1)
    corr_matrix = corr_data.corr()
    
    fig = px.imshow(corr_matrix,
                   text_auto=True,
                   color_continuous_scale='RdBu',
                   range_color=[-1, 1],
                   title="Correlation Matrix of Key Indicators")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Geospatial Analysis of Poverty in London")
    
    # Interactive Folium Map
    st.subheader("Interactive Borough-Level Poverty Map")
    
    # Prepare borough-level data
    df_merged = gdf.merge(df_la, left_on="DISTRICT", right_on="LA", how="inner")
    df_merged["Poverty_avg"] = df_merged[years].mean(axis=1)
    gdf_wards = gpd.GeoDataFrame(df_merged, geometry="geometry")
    
    gdf_boroughs = gdf_wards.dissolve(
        by="DISTRICT",
        aggfunc={col: 'mean' for col in years}
    ).reset_index()
    gdf_boroughs["Poverty_avg"] = gdf_wards.groupby("DISTRICT")["Poverty_avg"].mean().values
    
    # Create Folium map
    m = folium.Map(location=[51.5074, -0.1278], 
                  zoom_start=11, 
                  tiles=map_style,
                  control_scale=True)
    
    # Add choropleth layer
    folium.Choropleth(
        geo_data=gdf_boroughs,
        name="Poverty Average",
        data=gdf_boroughs,
        columns=["DISTRICT", "Poverty_avg"],
        key_on="feature.properties.DISTRICT",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="Average Poverty Cases (2018-2023)",
        highlight=True,
        bins=7
    ).add_to(m)
    
    # Add markers with borough names
    marker_cluster = MarkerCluster().add_to(m)
    
    for idx, row in gdf_boroughs.iterrows():
        centroid = row.geometry.centroid
        folium.Marker(
            location=[centroid.y, centroid.x],
            popup=f"<strong>{row['DISTRICT']}</strong><br>Avg Poverty: {row['Poverty_avg']:,.0f}",
            icon=None
        ).add_to(marker_cluster)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Display map
    folium_static(m, width=1000, height=600)
    
    # Borough ranking
    st.subheader("Borough Rankings by Poverty Levels")
    
    ranked_boroughs = gdf_boroughs.sort_values("Poverty_avg", ascending=False)[["DISTRICT", "Poverty_avg"]]
    ranked_boroughs["Rank"] = range(1, len(ranked_boroughs)+1)
    ranked_boroughs["Poverty_avg"] = ranked_boroughs["Poverty_avg"].round(0)
    
    fig = px.bar(ranked_boroughs.head(10),
                x="Poverty_avg",
                y="DISTRICT",
                orientation='h',
                color="Poverty_avg",
                color_continuous_scale="OrRd",
                title="Top 10 Boroughs by Poverty Levels (2018-2023 Avg)",
                labels={"Poverty_avg": "Average Poverty Cases", "DISTRICT": "Borough"})
    
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Advanced Analytics")
    
    # Time series decomposition
    st.subheader("Time Series Analysis")
    
    # Prepare time series data
    ts_data = df_la[df_la["LA"] == selected_borough].iloc[:, 1:].T
    ts_data.columns = ["Poverty Cases"]
    ts_data.index = pd.to_datetime(ts_data.index.str[:4], format='%Y')
    
    # Simple forecasting (using moving average)
    forecast_window = st.slider("Forecast Window (years)", 1, 5, 2)
    rolling_avg = ts_data.rolling(window=2).mean()
    forecast = pd.DataFrame({
        "Year": pd.date_range(start=ts_data.index[-1], periods=forecast_window+1, freq='YS')[1:],
        "Poverty Cases": [rolling_avg.iloc[-1,0]] * forecast_window
    }).set_index("Year")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts_data.index,
        y=ts_data["Poverty Cases"],
        mode='lines+markers',
        name='Historical Data',
        line=dict(color='#3498db')
    ))
    fig.add_trace(go.Scatter(
        x=forecast.index,
        y=forecast["Poverty Cases"],
        mode='lines+markers',
        name='Forecast (Moving Avg)',
        line=dict(color='#e74c3c', dash='dot')
    ))
    fig.update_layout(
        title=f"Poverty Cases in {selected_borough} with {forecast_window}-Year Forecast",
        xaxis_title="Year",
        yaxis_title="Poverty Cases",
        hovermode="x"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Multivariate analysis
    st.subheader("Multivariate Analysis")
    
    # Combine all datasets
    analysis_df = pd.DataFrame({
        'Borough': df_pobreza.index,
        'Child_Poverty_2022': df_pobreza[2022],
        'Unemployment_2022': df_desempleo[2022],
        'House_Price_Ratio_2022': df_precio_casa[2022]
    }).reset_index(drop = True)
    analysis_df = analysis_df.set_index('Borough')
    
    # Normalize data for radar chart
    scaler = MinMaxScaler()
    analysis_df[['Child_Poverty_2022', 'Unemployment_2022', 'House_Price_Ratio_2022']] = \
        scaler.fit_transform(analysis_df[['Child_Poverty_2022', 'Unemployment_2022', 'House_Price_Ratio_2022']])
    
    # Radar chart for selected borough
    #selected_data = analysis_df[analysis_df['Borough'] == selected_borough].iloc[0]
    selected_data = analysis_df.loc[selected_borough]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=selected_data[['Child_Poverty_2022', 'Unemployment_2022', 'House_Price_Ratio_2022']],
        theta=['Child Poverty', 'Unemployment', 'House Price Ratio'],
        fill='toself',
        name=selected_borough,
        line_color='#3498db'
    ))
    
        
    # Add London average for comparison
    london_avg = analysis_df.mean()
    fig.add_trace(go.Scatterpolar(
        r=london_avg[['Child_Poverty_2022', 'Unemployment_2022', 'House_Price_Ratio_2022']],
        theta=['Child Poverty', 'Unemployment', 'House Price Ratio'],
        fill='toself',
        name='London Average',
        line_color='#e74c3c'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title=f"Indicator Comparison: {selected_borough} vs London Average (2022)"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Policy impact simulation
    st.subheader("Policy Impact Simulation")
    
    col1, col2 = st.columns(2)
    with col1:
        unemployment_reduction = st.slider(
            "Unemployment Reduction Target (%)",
            0, 30, 10
        )
    with col2:
        housing_subsidy = st.slider(
            "Housing Affordability Improvement (%)",
            0, 30, 5
        )
    
    # Simple impact model
    current_poverty = df_pobreza.loc[selected_borough, 2022]
    estimated_impact = current_poverty * (1 - (unemployment_reduction/100 * 0.3 + housing_subsidy/100 * 0.2))
    
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=estimated_impact,
        number={'prefix': "", 'font': {'size': 40}},
        delta={'reference': current_poverty, 'relative': False, 'valueformat': '.0f'},
        title={"text": f"Projected Child Poverty in {selected_borough}"},
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    fig.update_layout(
        title="Policy Impact Simulation Results",
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div class="footer">
    <p> Data Science Project | Created with Python, GeoPandas, and Streamlit | Jocelyn Soto</p>
    <p>Last updated: {}</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d")), unsafe_allow_html=True)