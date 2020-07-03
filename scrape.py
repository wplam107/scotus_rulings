from scrape_functions import *

if __name__ == '__main__':
    # Build initial DataFrame
    url = 'https://en.wikipedia.org/wiki/1999_term_opinions_of_the_Supreme_Court_of_the_United_States'
    website_url = requests.get(url).text
    soup = BeautifulSoup(website_url, 'lxml')
    votes_by_case, justices = get_data(soup)
    df = pd.DataFrame.from_dict(votes_by_case)
    df.index = justices

    # Build and join DataFrames from remaining wikipedia tables
    years = list(range(2000, 2020))
    base_url = 'https://en.wikipedia.org/wiki/{}_term_opinions_of_the_Supreme_Court_of_the_United_States'
    urls = make_urls(base_url, years)

    for url in urls:
        website_url = requests.get(url).text
        soup = BeautifulSoup(website_url, 'lxml')
        votes_by_case, justices = get_data(soup)
        new_df = pd.DataFrame.from_dict(votes_by_case)
        new_df.index = justices
        df = df.join(new_df, how='outer')
    
    # Output as .csv file
    df.to_csv('scotus_rulings.csv')
    print('File Created')

