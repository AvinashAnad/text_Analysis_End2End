def extractfn(searchstring,datelimit):
    # remove previous files
    import os
    oldfiles = os.listdir()
    delfiles = [i for i in oldfiles if i.endswith('.png') or i.endswith('.pptx')or i.endswith('.csv')]
    [os.remove(i) for i in delfiles]
    
    import os
    import datetime
    datelimit = str(datetime.date.today() + datetime.timedelta(-int(datelimit))) #today minus 1 days
    # print (datelimit)
    starttime = datetime.datetime.now()
    try:
        os.system("pip install -r requirements.txt")
        os.system("cls")
    except: pass

    import tweepy
    import csv

    consumer_key = ""
    consumer_secret = ""
    access_token = ""
    access_token_secret  = "" 

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth,wait_on_rate_limit=True)

    file = open(str(searchstring)+'.csv', 'w')
    file.writelines('text'+'\n')
    file.close()

    csvFile = open(str(searchstring)+'.csv', 'a')

    csvWriter = csv.writer(csvFile)
    try:
        for tweet in tweepy.Cursor(api.search,q=searchstring,count=100,lang="en",since=datelimit).items():
            print (tweet.text)
            csvWriter.writerow([tweet.text.encode('utf-8')])
    except: pass
    try:
        os.system("pip freeze > requirements.txt")
    except: pass
    
    Endtime = datetime.datetime.now()
    print("Extraction started at " + str(starttime) + " ended at"+ str(Endtime))
    # 