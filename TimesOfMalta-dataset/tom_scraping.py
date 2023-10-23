import requests
import re
import csv

years = [2023, 2022, 2021, 2020]
# Category names
# Years - Years being taken into consideration (to divide csv files)
# Loops - Number of search pages per year
categories = {
    'national': {
        'years': years,
        'loops': [282, 361, 396, 406]
    },
    'world': {
        'years': years,
        'loops': [116, 157, 163, 140]
    },
    'opinion': {
        'years': years,
        'loops': [80, 110, 113, 92]
    },
    'sport': {
        'years': years,
        'loops': [197, 303, 259, 226]
    },
    'business': {
        'years': years,
        'loops': [86, 127, 162, 125]
    },
    'tech': {
        'years': years,
        'loops': [16, 18, 21, 8]
    },
    'entertainment': {
        'years': years,
        'loops': [67, 96, 93, 97]
    },
    'community': {
        'years': years,
        'loops': [98, 107, 135, 106]
    }
}

# Times of Malta Category IDs
category_ids = {
    'national': 1268,
    'world': 1280,
    'opinion': 1271,
    'sport': 1274,
    'business': 1256,
    'tech': 1277,
    'entertainment': 1262,
    'community': 1259
}


# Generates base URL for a given year and category
def tom_url(category, year):
    # Year 2023 only taking into consideration up to 22nd October 2023 (due to running date)
    if year == 2023:
        return f"https://timesofmalta.com/search?keywords=&fields%5B0%5D=title&fields%5B1%5D=body&tags%5B0%5D={category_ids[category]}&range=custom_range&from={year}-01-01&until=2023-10-22&sort=date&order=desc&author=0&page="
    else:
        return f"https://timesofmalta.com/search?keywords=&fields%5B0%5D=title&fields%5B1%5D=body&tags%5B0%5D={category_ids[category]}&range=custom_range&from={year}-01-01&until={year}-12-31&sort=date&order=desc&author=0&page="


# Updated function to scrape and save data for a specific category and year
def scrape(category, year, num_loops):
    base_url = tom_url(category, year)
    all_data = []  # Stores all data for the news category and year
    for i in range(num_loops):
        print(f"Category: {category}, Year: {year}, Page {i + 1}")
        url = f"{base_url}{i + 1}"
        response = requests.get(url)
        if response.status_code == 200:
            page_content = response.text
            article_pattern = re.compile(r'"headline":"(.*?)",.*?"description":"(.*?)",.*?"keywords":"(.*?)",.*?"articleBody":"(.*?)",.*?"datePublished":"(.*?)"')
            article_matches = article_pattern.findall(page_content)
            for title, subheading, keywords, article_body, date_published in article_matches:
                subheading = subheading.split('"publisher"')[0]
                all_data.append([title, subheading, keywords, article_body, date_published])
    # Storing data to csv file
    if all_data:
        filename = f'tom_{category}_{year}.csv'
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Title', 'Subheading', 'Keywords', 'Article Body', 'Date Published'])
            csv_writer.writerows(all_data)
    else:
        print(f"No data found/issue for Category: {category}, Year: {year}")

# Iterating through news categories and years
for category, info in categories.items():
    years = info['years']
    loops = info['loops']
    for i in range(len(years)):
        scrape(category, years[i], loops[i])
