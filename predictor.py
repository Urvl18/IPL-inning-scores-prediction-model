### Custom definitions and classes if any ###
import pandas as pd
import numpy 
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
# import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def predictRuns(testInput):
    prediction = 0
    rfg=RandomForestRegressor()
    le=preprocessing.LabelEncoder()
    
#     calling main data
    df=pd.read_csv("all_matches.csv")
    df.columns
    strikers=pd.DataFrame(df.striker.unique())
#     strikers.to_csv('D:/iitm ipl2020/strikers.csv')
    
    cols=['start_date','venue','innings','runs_off_bat','striker','bowler','match_id','ball','batting_team','bowling_team']
    df=df[cols]
    df['start_date']=pd.to_datetime(df['start_date'])
    df_pp=df[(df['ball']<6.0)&(df['start_date']>"2015-01-01")]

    
#     creating training data
    bowlers=pd.DataFrame(df_pp.bowler.unique())
    # bowlers.to_csv('D:/iitm ipl2020/bowler.csv')
    stadiums=pd.DataFrame(df_pp.venue.unique())
    # stadiums.to_csv('D:/iitm ipl2020/stdim.csv')
    ball_faced=pd.DataFrame(df_pp.groupby(['match_id','venue','innings','striker','bowler']).ball.count().reset_index())
    stadiums_0=pd.read_csv("stdim.csv")
    df_pp=pd.merge(df_pp,stadiums_0,on='venue')
    df_pp=pd.merge(df_pp.drop('ball',axis=1),ball_faced,on=['match_id','striker','bowler'])
    df_pp=df_pp.drop_duplicates()
    df_pp_00=df_pp.drop(['venue_x','innings_x','venue_y'],axis=1)
    
    
    
#     stadium data transformation

    stadiums_0['venue']=stadiums_0['venue'].replace(['MA Chidambaram Stadium, Chepauk, Chennai','MA Chidambaram Stadium','Arun Jaitley Stadium','Sawai Mansingh Stadium','Punjab Cricket Association IS Bindra Stadium','Dubai International Cricket Stadium'],'bowl')
    stadiums_0=stadiums_0[stadiums_0['venue'] != 'Sheikh Zayed Stadium']
    stadiums_0['venue']=stadiums_0['venue'].replace(['Wankhede Stadium, Mumbai','M.Chinnaswamy Stadium','MA Chidambaram Stadium, Chepauk','Eden Gardens','Wankhede Stadium','Sawai Mansingh Stadium','M Chinnaswamy Stadium','Rajiv Gandhi International Stadium','Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium','Sharjah Cricket Stadium'],'Bat')
    # stadiums_0.to_csv('D:/iitm ipl2020/stdim_type.csv')
    pitchtype_pref=pd.read_csv("stdim_type.csv")

    
    
    # bowler_type defined
    bowler_type=pd.read_csv("bwler.csv")
    df6=pd.merge(df_pp_00,bowler_type,on='bowler')

    # runs_after_6th_over
    runs=df6.groupby(['match_id','innings_y']).runs_off_bat.sum().reset_index()
    main_df=pd.merge(df6,runs,on=['match_id','innings_y'])

    # no_of_wicktes_in_inning
    df8=main_df.copy()
    no_wickets=df8.groupby(['match_id','innings_y']).striker.unique().reset_index()
    no_wickets['wickets']=[len(no_wickets['striker'].loc[i]) for i in range(len(no_wickets))]
    matchwise=pd.merge(no_wickets,runs,on=['match_id','innings_y'])
    
   # bowler_type_lable_encoding
    le_bowler=preprocessing.LabelEncoder()
    main_df['type']=le_bowler.fit_transform(main_df['type'])

    # spnner_to_pacer_ratio
    spinner_ratio=main_df.groupby(['match_id','innings_y','bowler']).type.mean().reset_index()
    spinner_ratio=spinner_ratio.groupby(['match_id','innings_y']).type.mean().reset_index()
    matchwise=pd.merge(matchwise,spinner_ratio,on=['match_id','innings_y'])
    matchwise=pd.merge(matchwise,main_df[['stdium','match_id']].drop_duplicates(),on='match_id')
    # matchwise['pacer_effect']=matchwise.runs_off_bat*(1-matchwise.type)
    # type_ratio=matchwise.groupby(['stdium']).type.std()
    # print(type_ratio)

    # pitch_type_preference
    le_stadium=preprocessing.LabelEncoder()
    matchwise=pd.merge(matchwise,pitchtype_pref,on=['stdium'])
    matchwise['std_type']=le_stadium.fit_transform(matchwise['std_type'])

    # stadium_target_encoding
#     te=ce.TargetEncoder(cols=['stdium'])
#     matchwise['stdium']=te.fit_transform(matchwise['stdium'],matchwise['runs_off_bat'])
    stdium_avg_score = matchwise.groupby('stdium').runs_off_bat.mean().reset_index()
    stdium_avg_score=stdium_avg_score.rename(columns = {'runs_off_bat':'std_scor'}, inplace = False)
    matchwise=pd.merge(matchwise,stdium_avg_score,on='stdium')
    matchwise['stdium']=matchwise['std_scor']
    

    # strike_rate_of_player
    strike_rate=(main_df.groupby(['striker','stdium','type']).runs_off_bat_x.sum()/main_df.groupby(['striker','stdium','type']).ball.sum()).reset_index()
    strike_rate=strike_rate.rename(columns = {0:'s_r'}, inplace = False)
    main_df.groupby(['striker','stdium']).runs_off_bat_x.sum().reset_index()[:20]

    # total_ball_faced_by_player
    player_ball_faced=main_df.groupby(['striker']).ball.sum().reset_index()

    # players_experience_weighted_strike_rate
    strike_rate=pd.merge(strike_rate,player_ball_faced,on='striker')
    strike_rate['exp_w_p']=strike_rate['s_r']*strike_rate['ball']
    df_bat=pd.merge(main_df,strike_rate,on=['striker','stdium','type'])
    exp_w_per=((df_bat.groupby(['match_id','innings_y']).exp_w_p.sum()/df_bat.groupby(['match_id','innings_y']).ball_y.sum())*100).reset_index()
    exp_w_per=exp_w_per.rename(columns = {0:'batmn_rate'}, inplace = False)
    matchwise=pd.merge(matchwise,exp_w_per,on=['match_id','innings_y'])

    # economy of bowler
    economy=(main_df.groupby(['bowler','stdium']).runs_off_bat_x.sum()/main_df.groupby(['bowler','stdium']).ball.sum()).reset_index()
    economy=economy.rename(columns = {0:'econ'}, inplace = False)
    # main_df.groupby(['bowler','stdium']).runs_off_bat_x.sum().reset_index()[:20]

    # total_ball_faced_by_player
    player_ball_delivered=main_df.groupby(['bowler']).ball.sum().reset_index()

    # players_experience_weighted_strike_rate
    economy=pd.merge(economy,player_ball_delivered,on='bowler')
    economy['exp_w_p']=economy['econ']*economy['ball']
    df_bwlr=pd.merge(main_df,economy,on=['bowler','stdium'])
    exp_w_per_bwlr=((df_bwlr.groupby(['match_id','innings_y']).exp_w_p.sum()/df_bwlr.groupby(['match_id','innings_y']).ball_y.sum())*100).reset_index()
    exp_w_per_bwlr=exp_w_per_bwlr.rename(columns = {0:'bwlr_rate'}, inplace = False)
    matchwise=pd.merge(matchwise,exp_w_per_bwlr,on=['match_id','innings_y'])

    
#     train test splitting
    features=matchwise[['innings_y','wickets','type','stdium','batmn_rate','bwlr_rate','std_type']]
    target=matchwise['runs_off_bat']
    x_tr, x_val, y_tr, y_val = train_test_split(features, target, train_size=0.75, test_size=0.25, random_state=0 )
    
    
#    model training 
    rfg=RandomForestRegressor(n_estimators=35,max_depth=50,random_state=2)
    rfg.fit(x_tr,y_tr)
    pred=rfg.predict(x_val)
    error=mean_absolute_error(y_val,pred)
    score=rfg.score(x_val,y_val)
    
#     input fila calling
    input_file=pd.read_csv(testInput)
    
    
#     input file processing
    i_stdium=pd.DataFrame(input_file['venue'])
    i_stdium=i_stdium.rename(columns = {'venue':'stdium'}, inplace = False)
    i_stdium=pd.merge(i_stdium,pitchtype_pref,on=['stdium'])
    i_stdium=pd.merge(i_stdium,stdium_avg_score,on=['stdium'])
    i_stdium=i_stdium.drop_duplicates()
    i_stdium['std_type']=le_stadium.transform(i_stdium['std_type'])
    xyz=int(i_stdium.std_type.mean())
    i_stdium['stdium']=i_stdium['std_scor']
    abc=float(i_stdium['stdium'])
#     i_stdium['stdium_score']=i_stdium['stdium']
    
    i_bowlers=pd.DataFrame(list(input_file['bowlers'])[0].split(','),columns=['bowler'])
    i_bowlers['stdium']=list(input_file['venue'])[0]
    b_type=bowler_type.drop('Unnamed: 0',axis=1)
    i_bowlers=pd.merge(i_bowlers,b_type,on='bowler',how='left')
    i_bowlers['type']=le_bowler.transform(i_bowlers['type'])
    i_bowlers=pd.merge(i_bowlers,economy,on=['bowler','stdium'],how='left')
    i_bowlers_wa=((i_bowlers['exp_w_p']).sum())/i_bowlers['ball'].sum()*100
    
    i_batsmen=input_file['batsmen']
    i_batsmen=pd.DataFrame(list(input_file['batsmen'])[0].split(','),columns=['striker'])
    i_batsmen['stdium']=list(input_file['venue'])[0]
    i_batsmen_pace=i_batsmen.copy()
    i_batsmen_spin=i_batsmen.copy()
    i_batsmen_pace['type']=0
    i_batsmen_spin['type']=0
    i_batsmen_pace=pd.merge(i_batsmen_pace,strike_rate,how='left',on=['striker','stdium','type'])
    i_batsmen_pace_wa=(i_batsmen_pace['exp_w_p'].sum())/i_batsmen_pace.ball.sum()*100
    i_batsmen_spin=pd.merge(i_batsmen_spin,strike_rate,on=['striker','stdium','type'])
    i_batsmen_spin_wa=(i_batsmen_spin['exp_w_p'].sum())/i_batsmen_spin.ball.sum()*100
    
    input_test=pd.DataFrame(columns=x_tr.columns)
    input_test['innings_y']=input_file['innings']
    input_test['wickets']=len(i_batsmen)
    input_test.at[0, 'type'] =(i_bowlers['type'].mean())
    input_test['std_type']=xyz
    input_test['stdium']=abc
    input_test['batmn_rate']=(((input_test['type'])*i_batsmen_spin_wa)+((1-(input_test['type']))*i_batsmen_pace_wa))
    input_test['bwlr_rate']=i_bowlers_wa
    
    for i in list(input_test.columns):
        if list(input_test[i].isnull())==[True]:
            input_test[i]=x_tr[i].mean()
    input_test
    
    prediction=round((float(rfg.predict(input_test))+40)/2)
    
    ### Your Code Here ###
    return prediction

