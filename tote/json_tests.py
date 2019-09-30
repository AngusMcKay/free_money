
import json

with open('/home/angus/analytics/product/dev_work/tote/json_yarmouth.json') as f:
    json_data = f.read()

parsed_json = json.loads(json_data)

parsed_json['meeting']['meetingPools'][0]['legs'][1]['breakdown']['horseStakes']

parsed_json['meeting']['races'][0]['racePools'][0]['dividendGuides']



### NOTES
# Within meetingPools there are two or three jsons for each venue (it is three if that venue has the day's Jackpot)
# Placepot is usually the first json, and can be identified because it has 'name' 1 and 6 'legs'
# Quadpot can be identified because it has 4 'legs' (this is the second json unless the Jackpot is at the venue, in which case it is third)
# Jackpot can be identified because it has 6 'legs' and 'name' 2 (will be second json if it exists for the venue that day)
# each has grossTotal, netTotal and legs, and the netTotal is shown as the pool size at top of the page
# each leg has the leg number and breakdown, which in turn has remainingStake and horseStakes
# horseStakes are what is shown in the pie chart segments, and remainingStake is shown as units in the centre of the pie

meeting_pools = parsed_json['meeting']['meetingPools']
def get_jackpot_data(meetingpool):
    if(len(meeting_pools)<3):
        print("no jackpot at meeting")
        return
    
    for pool in meeting_pools:
        if pool['name']==2:
            jackpot_details = pool
            break
    
    return jackpot_details

jackpot_details = get_jackpot_data(meeting_pools)
jackpot_details['netTotal'] # pool size at top of screen
jackpot_details['legs'][0]['breakdown'] # contains details needed for table and pie chart for first race (note, last horse numbered -1 is for the unnamed favourite)
jackpot_details['legs'][1]['breakdown'] # contains details needed for table and pie chart for second race
jackpot_details['dividend'] # gives results once available

def get_placepot_data(meetingpool):
    for pool in meeting_pools:
        if pool['name']==1:
            placepot_details = pool
            break
    
    return placepot_details

placepot_details = get_placepot_data(meeting_pools)
placepot_details['netTotal'] # pool size at top of screen
placepot_details['legs'][0]['breakdown'] # details for first race
placepot_details['dividend'] # gives results once available

def get_quadpot_data(meetingpool):
    for pool in meeting_pools:
        if len(pool['legs'])==4:
            quadpot_details = pool
            break
    
    return quadpot_details

quadpot_details = get_quadpot_data(meeting_pools)
quadpot_details['netTotal'] # pool size at top of screen
quadpot_details['legs'][0]['breakdown'] # details for first race
quadpot_details['dividend'] # gives results once available


# Details for wins and placings are in the Races section of the json
# within races have each race
# for each race have distance, name, status, time, toteSubstitute, horses, racePools and result
# for each horse have number, name, silkUrl, status, previous prices, position, currentPrice
# racePools 0 is winner, 1 is placed, 2 is forecast, 3 is tricast, 4 is name 2 to come in top 3 (not sure if order could change)
# in each racePool have: grossTotal, deductionPercentage, netTotal, dividendGuides, dividends
# dividendGuides gives the payout per unit for each horse, dividends gives the actual payout to winning horse(s) and the units on that horse

