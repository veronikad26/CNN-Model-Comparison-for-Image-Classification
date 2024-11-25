#the dataset is downloaded from kaggle and the CSV file is uploaded to Google Colab

#basic information about the dataset
import pandas as pd
df=pd.read_csv('outbreaks.csv')
print(df.head())
print(df.describe())

#QUESTION-1
import matplotlib.pyplot as plt
df=pd.read_csv("outbreaks.csv")
print(df["Year"])
outbreaks_per_year=df.groupby('Year').size()
print(outbreaks_per_year)
plt.figure(figsize=(10, 6))
outbreaks_per_year.plot(kind='line', marker='o')
plt.title('Trend of Foodborne Disease Outbreaks Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Outbreaks')
plt.grid(True)
plt.show()
trend='increasing' if outbreaks_per_year.iloc[-1] > outbreaks_per_year.iloc[0] else 'decreasing'
print(f"Foodborne disease outbreaks are {trend}.")

#QUESTION-2
contaminant_summary=df.groupby('Species')[['Illnesses', 'Hospitalizations', 'Fatalities']].sum()
most_illnesses=contaminant_summary['Illnesses'].idxmax()
most_hospitalizations=contaminant_summary['Hospitalizations'].idxmax()
most_deaths=contaminant_summary['Fatalities'].idxmax()
print(f"Contaminant responsible for the most illnesses: {most_illnesses}")
print(f"Contaminant responsible for the most hospitalizations: {most_hospitalizations}")
print(f"Contaminant responsible for the most deaths: {most_deaths}")
contaminant_summary.sort_values('Illnesses', ascending=False).head(10).plot(kind='bar',figsize=(14,7))
plt.title('Top 10 Species Responsible for Illnesses, Hospitalizations, and Deaths')
plt.xlabel('Contaminant')
plt.ylabel('Counts')
plt.grid(True)
plt.show()

#QUESTION-3
location_risk=df.groupby('Location')['Illnesses'].sum()
highest_risk_location=location_risk.idxmax()
print(f"The location for food preparation that poses the greatest risk of foodborne illness: {highest_risk_location}")
location_risk.sort_values(ascending=False).head(10).plot(kind='bar',figsize=(14,7))
plt.title('Top 10 Locations Posing Greatest Risk of Foodborne Illness')
plt.xlabel('Location')
plt.ylabel('Number of Illnesses')
plt.grid(True)
plt.show()
