from mpi4py import MPI
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup as bs4

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# kumpulkan data benua
def collect_continents():
    if rank == 0:
        continents_page = requests.get("https://simple.wikipedia.org/wiki/List_of_countries_by_continents").text
        continents_countries_soup = bs4(continents_page, "lxml")
        continents = continents_countries_soup.find_all('h2' > 'span', {"class": "mw-headline"})

        unwanted_words = ["Antarctica", "References", "Other websites"]
        target_continents = [continent.text for continent in continents if continent.text not in unwanted_words]

        ol_html = continents_countries_soup.find_all('ol')
        all_countries = [countries.find_all('li', {"class": None, "id": None}) for countries in ol_html]

        countries_in_continents = []
        for items in all_countries:
            countries = []
            if items:
                for country in items:
                    countries = [country.find('a').text for country in items if country.find('a')]
                countries_in_continents.append(countries)

        return countries_in_continents, target_continents

    else:
        return None, None

# kumpulkan data skor kebahagiaan
def collect_happiness_scores():
    if rank == 0:
        countries_score_page = requests.get("https://en.wikipedia.org/wiki/World_Happiness_Report#2020_report")
        countries_score_soup = bs4(countries_score_page.content, 'lxml')
        countries_score_table = countries_score_soup.find_all('table', {'class': 'wikitable sortable'})[3]
        countries_score_df = pd.read_html(str(countries_score_table))
        return countries_score_df[0]

    else:
        return None

countries_in_continents, target_continents = comm.bcast(collect_continents(), root=0)

countries_score_df = comm.bcast(collect_happiness_scores(), root=0)

# gabung data benua dan negara
if rank == 0:
    countries_continent_category_df = pd.DataFrame(
        zip(countries_in_continents, target_continents), columns=['Country', 'Continent'])

    countries_continent_category_df = countries_continent_category_df.explode(
        'Country').reset_index(drop=True)
    print("DATA BENUA DAN NEGARA")
    print(countries_continent_category_df)
    print()

# gabung data negara dan skor kebahagiaan
if rank == 0:
    print("DATA SKOR KEBAHAGIAAN")
    print(countries_score_df)
    print()

    print("WORLD HAPPINESS REPORT")
    countries_score_df = countries_score_df.rename(columns={"Country or region": "Country"})
    merged_df = pd.merge(countries_score_df, countries_continent_category_df, how='inner', on='Country')
    merged_df.to_csv('final_result.csv')
    print(merged_df)

# Sinkronisasi proses sebelum melanjutkan
comm.Barrier()

# tampilkan hasil
if rank == 0:
    print("TAMPILKAN DIAGRAM BATANG, HISTOGRAM, HEATMAP, DAN SCATTER PLOT")
    final_result_df_score_index = pd.read_csv('final_result.csv', index_col=2)
    ax = final_result_df_score_index['Score'].plot(kind='bar', figsize=(30, 5), title="Happiness Score For Each Country")
    ax.set_ylabel("Happiness Score")
    print()

    plt.figure(figsize=(45, 15))
    plt.title("Histogram of Number of Countries Within A Range of Happiness Score")
    plt.xlabel("Happiness Score")
    plt.ylabel("Number of Countries")
    plt.hist(final_result_df_score_index['Score'], bins=9)
    plt.show()
    print()

    heatmap_df = merged_df.drop(['Overall rank', 'Country', 'Continent'], axis=1)
    ax = sns.heatmap(heatmap_df.corr(), annot=True, fmt='.2f', cmap='Blues')
    print()

    sns.lmplot(x='GDP per capita', y='Score', data=merged_df, fit_reg=False)
    sns.lmplot(x='GDP per capita', y='Score', data=merged_df, fit_reg=False, hue='Continent')
