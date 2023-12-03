import pandas as pd

# 데이터 불러오기
data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

# 2015 - 2018년 top 10
def top10_year(df, p_year, p_stat):
    result = {}
    for year in p_year:
        result[year] = {}
        for stat in p_stat:
            result[year][stat] = df[df['year'] == year].nlargest(10, stat)[['batter_name', stat]]
    return result

# 포지션별 WAR 선수 (2018)
def top10_position(df, year):
    top10 = {}
    position = df['cp'].unique()
    for pos in position:
        players_in_position = df[(df['year'] == year) & (df['cp'] == pos)]
        top_player = players_in_position.nlargest(1, 'war')[['batter_name', 'war']]
        if top_player.empty: continue
        top10[pos] = top_player
    return top10

# corr 최대값 찾기 (salary)
def corr_highest_salary(df):
    correlation = df[['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG', 'salary']].corr()['salary']
    return correlation.drop('salary').abs().idxmax(), correlation

# 함수 적용
year = [2015, 2016, 2017, 2018]
stat = ['H', 'avg', 'HR', 'OBP']
top10_year_stat = top10_year(data_df, year, stat)
top10_position_2018 = top10_position(data_df, 2018)
corr_stat, corr_salary = corr_highest_salary(data_df)

# 결과
print("2015년 안타(H) 상위 10명 선수: ")
print(top10_year_stat[2015]['H'])
print("\n2018년 포지션별 최고 WAR 선수: ")
print(top10_position_2018)
print("\n연봉과 가장 높은 상관관계를 가진 통계치: ", corr_stat)