
USE tennis;

SELECT *
FROM mens
LIMIT 100;

SELECT *
FROM men_players_historic
LIMIT 100;


-- order data with row numbers
WITH

	historic_ordered AS
	(SELECT *, ROW_NUMBER() OVER(PARTITION BY player, clay, best_of ORDER BY date ASC) AS rn
	FROM men_players_historic),
    
    mens_ordered AS
    (SELECT *
    FROM mens m
    LEFT JOIN (SELECT player AS p1, opponent AS o1, date AS dt1, rn AS p1_rn FROM historic_ordered) h1
    ON m.player1 = h1.p1 AND m.player2 = h1.o1 AND m.date = h1.dt1
    LEFT JOIN (SELECT player AS p2, opponent AS o2, date AS dt2, rn AS p2_rn FROM historic_ordered) h2
    ON m.player2 = h2.p2 AND m.player1 = h2.o2 AND m.date = h2.dt2)



-- overall query
SELECT m.best_of, m.date, m.carpet, m.clay, m.grass, m.hard, m.p1Rank, m.b365P1, m.b365P1prob, m.p2Rank, m.b365P2, m.b365P2prob, m.b365bookiesgain, m.player1Wins,
		
        p11.playerWins AS p11Win, p11.best_of AS p11Best_of, p11.carpet AS p11Carpet, p11.clay AS p11Clay, p11.grass AS p11Grass, p11.hard AS p11Hard,
        p11.playerRank AS p11PlayerRank, p11.playersets AS p11Playersets, p11.playergames1 AS p11Playergames1, p11.playergames2 AS p11Playergames2,
        p11.playergames3 AS p11Playergames3, p11.playergames4 AS p11Playergames4, p11.playergames5 AS p11Playergames5, p11.b365playerprob AS p11Playerprob,
        p11.opponentRank AS p11OpponentRank, p11.opponentsets AS p11Opponentsets, p11.opponentgames1 AS p11Opponentgames1, p11.opponentgames2 AS p11Opponentgames2,
        p11.opponentgames3 AS p11Opponentgames3, p11.opponentgames4 AS p11Opponentgames4, p11.opponentgames5 AS p11Opponentgames5, p11.b365opponentprob AS p11Opponentprob,
		
        p12.playerWins AS p12Win, p12.best_of AS p12Best_of, p12.carpet AS p12Carpet, p12.clay AS p12Clay, p12.grass AS p12Grass, p12.hard AS p12Hard,
        p12.playerRank AS p12PlayerRank, p12.playersets AS p12Playersets, p12.playergames1 AS p12Playergames1, p12.playergames2 AS p12Playergames2,
        p12.playergames3 AS p12Playergames3, p12.playergames4 AS p12Playergames4, p12.playergames5 AS p12Playergames5, p12.b365playerprob AS p12Playerprob,
        p12.opponentRank AS p12OpponentRank, p12.opponentsets AS p12Opponentsets, p12.opponentgames1 AS p12Opponentgames1, p12.opponentgames2 AS p12Opponentgames2,
        p12.opponentgames3 AS p12Opponentgames3, p12.opponentgames4 AS p12Opponentgames4, p12.opponentgames5 AS p12Opponentgames5, p12.b365opponentprob AS p12Opponentprob,
        
		p13.playerWins AS p13Win, p13.best_of AS p13Best_of, p13.carpet AS p13Carpet, p13.clay AS p13Clay, p13.grass AS p13Grass, p13.hard AS p13Hard,
        p13.playerRank AS p13PlayerRank, p13.playersets AS p13Playersets, p13.playergames1 AS p13Playergames1, p13.playergames2 AS p13Playergames2,
        p13.playergames3 AS p13Playergames3, p13.playergames4 AS p13Playergames4, p13.playergames5 AS p13Playergames5, p13.b365playerprob AS p13Playerprob,
        p13.opponentRank AS p13OpponentRank, p13.opponentsets AS p13Opponentsets, p13.opponentgames1 AS p13Opponentgames1, p13.opponentgames2 AS p13Opponentgames2,
        p13.opponentgames3 AS p13Opponentgames3, p13.opponentgames4 AS p13Opponentgames4, p13.opponentgames5 AS p13Opponentgames5, p13.b365opponentprob AS p13Opponentprob,

		p14.playerWins AS p14Win, p14.best_of AS p14Best_of, p14.carpet AS p14Carpet, p14.clay AS p14Clay, p14.grass AS p14Grass, p14.hard AS p14Hard,
        p14.playerRank AS p14PlayerRank, p14.playersets AS p14Playersets, p14.playergames1 AS p14Playergames1, p14.playergames2 AS p14Playergames2,
        p14.playergames3 AS p14Playergames3, p14.playergames4 AS p14Playergames4, p14.playergames5 AS p14Playergames5, p14.b365playerprob AS p14Playerprob,
        p14.opponentRank AS p14OpponentRank, p14.opponentsets AS p14Opponentsets, p14.opponentgames1 AS p14Opponentgames1, p14.opponentgames2 AS p14Opponentgames2,
        p14.opponentgames3 AS p14Opponentgames3, p14.opponentgames4 AS p14Opponentgames4, p14.opponentgames5 AS p14Opponentgames5, p14.b365opponentprob AS p14Opponentprob,

		p15.playerWins AS p15Win, p15.best_of AS p15Best_of, p15.carpet AS p15Carpet, p15.clay AS p15Clay, p15.grass AS p15Grass, p15.hard AS p15Hard,
        p15.playerRank AS p15PlayerRank, p15.playersets AS p15Playersets, p15.playergames1 AS p15Playergames1, p15.playergames2 AS p15Playergames2,
        p15.playergames3 AS p15Playergames3, p15.playergames4 AS p15Playergames4, p15.playergames5 AS p15Playergames5, p15.b365playerprob AS p15Playerprob,
        p15.opponentRank AS p15OpponentRank, p15.opponentsets AS p15Opponentsets, p15.opponentgames1 AS p15Opponentgames1, p15.opponentgames2 AS p15Opponentgames2,
        p15.opponentgames3 AS p15Opponentgames3, p15.opponentgames4 AS p15Opponentgames4, p15.opponentgames5 AS p15Opponentgames5, p15.b365opponentprob AS p15Opponentprob,

		p16.playerWins AS p16Win, p16.best_of AS p16Best_of, p16.carpet AS p16Carpet, p16.clay AS p16Clay, p16.grass AS p16Grass, p16.hard AS p16Hard,
        p16.playerRank AS p16PlayerRank, p16.playersets AS p16Playersets, p16.playergames1 AS p16Playergames1, p16.playergames2 AS p16Playergames2,
        p16.playergames3 AS p16Playergames3, p16.playergames4 AS p16Playergames4, p16.playergames5 AS p16Playergames5, p16.b365playerprob AS p16Playerprob,
        p16.opponentRank AS p16OpponentRank, p16.opponentsets AS p16Opponentsets, p16.opponentgames1 AS p16Opponentgames1, p16.opponentgames2 AS p16Opponentgames2,
        p16.opponentgames3 AS p16Opponentgames3, p16.opponentgames4 AS p16Opponentgames4, p16.opponentgames5 AS p16Opponentgames5, p16.b365opponentprob AS p16Opponentprob,

		p17.playerWins AS p17Win, p17.best_of AS p17Best_of, p17.carpet AS p17Carpet, p17.clay AS p17Clay, p17.grass AS p17Grass, p17.hard AS p17Hard,
        p17.playerRank AS p17PlayerRank, p17.playersets AS p17Playersets, p17.playergames1 AS p17Playergames1, p17.playergames2 AS p17Playergames2,
        p17.playergames3 AS p17Playergames3, p17.playergames4 AS p17Playergames4, p17.playergames5 AS p17Playergames5, p17.b365playerprob AS p17Playerprob,
        p17.opponentRank AS p17OpponentRank, p17.opponentsets AS p17Opponentsets, p17.opponentgames1 AS p17Opponentgames1, p17.opponentgames2 AS p17Opponentgames2,
        p17.opponentgames3 AS p17Opponentgames3, p17.opponentgames4 AS p17Opponentgames4, p17.opponentgames5 AS p17Opponentgames5, p17.b365opponentprob AS p17Opponentprob,

		p18.playerWins AS p18Win, p18.best_of AS p18Best_of, p18.carpet AS p18Carpet, p18.clay AS p18Clay, p18.grass AS p18Grass, p18.hard AS p18Hard,
        p18.playerRank AS p18PlayerRank, p18.playersets AS p18Playersets, p18.playergames1 AS p18Playergames1, p18.playergames2 AS p18Playergames2,
        p18.playergames3 AS p18Playergames3, p18.playergames4 AS p18Playergames4, p18.playergames5 AS p18Playergames5, p18.b365playerprob AS p18Playerprob,
        p18.opponentRank AS p18OpponentRank, p18.opponentsets AS p18Opponentsets, p18.opponentgames1 AS p18Opponentgames1, p18.opponentgames2 AS p18Opponentgames2,
        p18.opponentgames3 AS p18Opponentgames3, p18.opponentgames4 AS p18Opponentgames4, p18.opponentgames5 AS p18Opponentgames5, p18.b365opponentprob AS p18Opponentprob,

		p19.playerWins AS p19Win, p19.best_of AS p19Best_of, p19.carpet AS p19Carpet, p19.clay AS p19Clay, p19.grass AS p19Grass, p19.hard AS p19Hard,
        p19.playerRank AS p19PlayerRank, p19.playersets AS p19Playersets, p19.playergames1 AS p19Playergames1, p19.playergames2 AS p19Playergames2,
        p19.playergames3 AS p19Playergames3, p19.playergames4 AS p19Playergames4, p19.playergames5 AS p19Playergames5, p19.b365playerprob AS p19Playerprob,
        p19.opponentRank AS p19OpponentRank, p19.opponentsets AS p19Opponentsets, p19.opponentgames1 AS p19Opponentgames1, p19.opponentgames2 AS p19Opponentgames2,
        p19.opponentgames3 AS p19Opponentgames3, p19.opponentgames4 AS p19Opponentgames4, p19.opponentgames5 AS p19Opponentgames5, p19.b365opponentprob AS p19Opponentprob,

		p110.playerWins AS p110Win, p110.best_of AS p110Best_of, p110.carpet AS p110Carpet, p110.clay AS p110Clay, p110.grass AS p110Grass, p110.hard AS p110Hard,
        p110.playerRank AS p110PlayerRank, p110.playersets AS p110Playersets, p110.playergames1 AS p110Playergames1, p110.playergames2 AS p110Playergames2,
        p110.playergames3 AS p110Playergames3, p110.playergames4 AS p110Playergames4, p110.playergames5 AS p110Playergames5, p110.b365playerprob AS p110Playerprob,
        p110.opponentRank AS p110OpponentRank, p110.opponentsets AS p110Opponentsets, p110.opponentgames1 AS p110Opponentgames1, p110.opponentgames2 AS p110Opponentgames2,
        p110.opponentgames3 AS p110Opponentgames3, p110.opponentgames4 AS p110Opponentgames4, p110.opponentgames5 AS p110Opponentgames5, p110.b365opponentprob AS p110Opponentprob,

        p21.playerWins AS p21Win, p21.best_of AS p21Best_of, p21.carpet AS p21Carpet, p21.clay AS p21Clay, p21.grass AS p21Grass, p21.hard AS p21Hard,
        p21.playerRank AS p21PlayerRank, p21.playersets AS p21Playersets, p21.playergames1 AS p21Playergames1, p21.playergames2 AS p21Playergames2,
        p21.playergames3 AS p21Playergames3, p21.playergames4 AS p21Playergames4, p21.playergames5 AS p21Playergames5, p21.b365playerprob AS p21Playerprob,
        p21.opponentRank AS p21OpponentRank, p21.opponentsets AS p21Opponentsets, p21.opponentgames1 AS p21Opponentgames1, p21.opponentgames2 AS p21Opponentgames2,
        p21.opponentgames3 AS p21Opponentgames3, p21.opponentgames4 AS p21Opponentgames4, p21.opponentgames5 AS p21Opponentgames5, p21.b365opponentprob AS p21Opponentprob,
		
        p22.playerWins AS p22Win, p22.best_of AS p22Best_of, p22.carpet AS p22Carpet, p22.clay AS p22Clay, p22.grass AS p22Grass, p22.hard AS p22Hard,
        p22.playerRank AS p22PlayerRank, p22.playersets AS p22Playersets, p22.playergames1 AS p22Playergames1, p22.playergames2 AS p22Playergames2,
        p22.playergames3 AS p22Playergames3, p22.playergames4 AS p22Playergames4, p22.playergames5 AS p22Playergames5, p22.b365playerprob AS p22Playerprob,
        p22.opponentRank AS p22OpponentRank, p22.opponentsets AS p22Opponentsets, p22.opponentgames1 AS p22Opponentgames1, p22.opponentgames2 AS p22Opponentgames2,
        p22.opponentgames3 AS p22Opponentgames3, p22.opponentgames4 AS p22Opponentgames4, p22.opponentgames5 AS p22Opponentgames5, p22.b365opponentprob AS p22Opponentprob,
        
		p23.playerWins AS p23Win, p23.best_of AS p23Best_of, p23.carpet AS p23Carpet, p23.clay AS p23Clay, p23.grass AS p23Grass, p23.hard AS p23Hard,
        p23.playerRank AS p23PlayerRank, p23.playersets AS p23Playersets, p23.playergames1 AS p23Playergames1, p23.playergames2 AS p23Playergames2,
        p23.playergames3 AS p23Playergames3, p23.playergames4 AS p23Playergames4, p23.playergames5 AS p23Playergames5, p23.b365playerprob AS p23Playerprob,
        p23.opponentRank AS p23OpponentRank, p23.opponentsets AS p23Opponentsets, p23.opponentgames1 AS p23Opponentgames1, p23.opponentgames2 AS p23Opponentgames2,
        p23.opponentgames3 AS p23Opponentgames3, p23.opponentgames4 AS p23Opponentgames4, p23.opponentgames5 AS p23Opponentgames5, p23.b365opponentprob AS p23Opponentprob,

		p24.playerWins AS p24Win, p24.best_of AS p24Best_of, p24.carpet AS p24Carpet, p24.clay AS p24Clay, p24.grass AS p24Grass, p24.hard AS p24Hard,
        p24.playerRank AS p24PlayerRank, p24.playersets AS p24Playersets, p24.playergames1 AS p24Playergames1, p24.playergames2 AS p24Playergames2,
        p24.playergames3 AS p24Playergames3, p24.playergames4 AS p24Playergames4, p24.playergames5 AS p24Playergames5, p24.b365playerprob AS p24Playerprob,
        p24.opponentRank AS p24OpponentRank, p24.opponentsets AS p24Opponentsets, p24.opponentgames1 AS p24Opponentgames1, p24.opponentgames2 AS p24Opponentgames2,
        p24.opponentgames3 AS p24Opponentgames3, p24.opponentgames4 AS p24Opponentgames4, p24.opponentgames5 AS p24Opponentgames5, p24.b365opponentprob AS p24Opponentprob,

		p25.playerWins AS p25Win, p25.best_of AS p25Best_of, p25.carpet AS p25Carpet, p25.clay AS p25Clay, p25.grass AS p25Grass, p25.hard AS p25Hard,
        p25.playerRank AS p25PlayerRank, p25.playersets AS p25Playersets, p25.playergames1 AS p25Playergames1, p25.playergames2 AS p25Playergames2,
        p25.playergames3 AS p25Playergames3, p25.playergames4 AS p25Playergames4, p25.playergames5 AS p25Playergames5, p25.b365playerprob AS p25Playerprob,
        p25.opponentRank AS p25OpponentRank, p25.opponentsets AS p25Opponentsets, p25.opponentgames1 AS p25Opponentgames1, p25.opponentgames2 AS p25Opponentgames2,
        p25.opponentgames3 AS p25Opponentgames3, p25.opponentgames4 AS p25Opponentgames4, p25.opponentgames5 AS p25Opponentgames5, p25.b365opponentprob AS p25Opponentprob,

		p26.playerWins AS p26Win, p26.best_of AS p26Best_of, p26.carpet AS p26Carpet, p26.clay AS p26Clay, p26.grass AS p26Grass, p26.hard AS p26Hard,
        p26.playerRank AS p26PlayerRank, p26.playersets AS p26Playersets, p26.playergames1 AS p26Playergames1, p26.playergames2 AS p26Playergames2,
        p26.playergames3 AS p26Playergames3, p26.playergames4 AS p26Playergames4, p26.playergames5 AS p26Playergames5, p26.b365playerprob AS p26Playerprob,
        p26.opponentRank AS p26OpponentRank, p26.opponentsets AS p26Opponentsets, p26.opponentgames1 AS p26Opponentgames1, p26.opponentgames2 AS p26Opponentgames2,
        p26.opponentgames3 AS p26Opponentgames3, p26.opponentgames4 AS p26Opponentgames4, p26.opponentgames5 AS p26Opponentgames5, p26.b365opponentprob AS p26Opponentprob,

		p27.playerWins AS p27Win, p27.best_of AS p27Best_of, p27.carpet AS p27Carpet, p27.clay AS p27Clay, p27.grass AS p27Grass, p27.hard AS p27Hard,
        p27.playerRank AS p27PlayerRank, p27.playersets AS p27Playersets, p27.playergames1 AS p27Playergames1, p27.playergames2 AS p27Playergames2,
        p27.playergames3 AS p27Playergames3, p27.playergames4 AS p27Playergames4, p27.playergames5 AS p27Playergames5, p27.b365playerprob AS p27Playerprob,
        p27.opponentRank AS p27OpponentRank, p27.opponentsets AS p27Opponentsets, p27.opponentgames1 AS p27Opponentgames1, p27.opponentgames2 AS p27Opponentgames2,
        p27.opponentgames3 AS p27Opponentgames3, p27.opponentgames4 AS p27Opponentgames4, p27.opponentgames5 AS p27Opponentgames5, p27.b365opponentprob AS p27Opponentprob,

		p28.playerWins AS p28Win, p28.best_of AS p28Best_of, p28.carpet AS p28Carpet, p28.clay AS p28Clay, p28.grass AS p28Grass, p28.hard AS p28Hard,
        p28.playerRank AS p28PlayerRank, p28.playersets AS p28Playersets, p28.playergames1 AS p28Playergames1, p28.playergames2 AS p28Playergames2,
        p28.playergames3 AS p28Playergames3, p28.playergames4 AS p28Playergames4, p28.playergames5 AS p28Playergames5, p28.b365playerprob AS p28Playerprob,
        p28.opponentRank AS p28OpponentRank, p28.opponentsets AS p28Opponentsets, p28.opponentgames1 AS p28Opponentgames1, p28.opponentgames2 AS p28Opponentgames2,
        p28.opponentgames3 AS p28Opponentgames3, p28.opponentgames4 AS p28Opponentgames4, p28.opponentgames5 AS p28Opponentgames5, p28.b365opponentprob AS p28Opponentprob,

		p29.playerWins AS p29Win, p29.best_of AS p29Best_of, p29.carpet AS p29Carpet, p29.clay AS p29Clay, p29.grass AS p29Grass, p29.hard AS p29Hard,
        p29.playerRank AS p29PlayerRank, p29.playersets AS p29Playersets, p29.playergames1 AS p29Playergames1, p29.playergames2 AS p29Playergames2,
        p29.playergames3 AS p29Playergames3, p29.playergames4 AS p29Playergames4, p29.playergames5 AS p29Playergames5, p29.b365playerprob AS p29Playerprob,
        p29.opponentRank AS p29OpponentRank, p29.opponentsets AS p29Opponentsets, p29.opponentgames1 AS p29Opponentgames1, p29.opponentgames2 AS p29Opponentgames2,
        p29.opponentgames3 AS p29Opponentgames3, p29.opponentgames4 AS p29Opponentgames4, p29.opponentgames5 AS p29Opponentgames5, p29.b365opponentprob AS p29Opponentprob,

		p210.playerWins AS p210Win, p210.best_of AS p210Best_of, p210.carpet AS p210Carpet, p210.clay AS p210Clay, p210.grass AS p210Grass, p210.hard AS p210Hard,
        p210.playerRank AS p210PlayerRank, p210.playersets AS p210Playersets, p210.playergames1 AS p210Playergames1, p210.playergames2 AS p210Playergames2,
        p210.playergames3 AS p210Playergames3, p210.playergames4 AS p210Playergames4, p210.playergames5 AS p210Playergames5, p210.b365playerprob AS p210Playerprob,
        p210.opponentRank AS p210OpponentRank, p210.opponentsets AS p210Opponentsets, p210.opponentgames1 AS p210Opponentgames1, p210.opponentgames2 AS p210Opponentgames2,
        p210.opponentgames3 AS p210Opponentgames3, p210.opponentgames4 AS p210Opponentgames4, p210.opponentgames5 AS p210Opponentgames5, p210.b365opponentprob AS p210Opponentprob

-- INTO OUTFILE '/var/lib/mysql-files/mens_data_sql_processed.csv'
-- FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
-- LINES TERMINATED BY '\n'


FROM mens_ordered m
-- join previous games on the same court (clay vs non-clay) and set length (i.e. slam vs non-slam)
-- player 1
LEFT JOIN historic_ordered p11
ON m.player1 = p11.player AND m.p1_rn = p11.rn+1 AND m.clay = p11.clay AND m.best_of = p11.best_of
LEFT JOIN historic_ordered p12
ON m.player1 = p12.player AND m.p1_rn = p12.rn+2 AND m.clay = p12.clay AND m.best_of = p12.best_of
LEFT JOIN historic_ordered p13
ON m.player1 = p13.player AND m.p1_rn = p13.rn+3 AND m.clay = p13.clay AND m.best_of = p13.best_of
LEFT JOIN historic_ordered p14
ON m.player1 = p14.player AND m.p1_rn = p14.rn+4 AND m.clay = p14.clay AND m.best_of = p14.best_of
LEFT JOIN historic_ordered p15
ON m.player1 = p15.player AND m.p1_rn = p15.rn+5 AND m.clay = p15.clay AND m.best_of = p15.best_of
LEFT JOIN historic_ordered p16
ON m.player1 = p16.player AND m.p1_rn = p16.rn+6 AND m.clay = p16.clay AND m.best_of = p16.best_of
LEFT JOIN historic_ordered p17
ON m.player1 = p17.player AND m.p1_rn = p17.rn+7 AND m.clay = p17.clay AND m.best_of = p17.best_of
LEFT JOIN historic_ordered p18
ON m.player1 = p18.player AND m.p1_rn = p18.rn+8 AND m.clay = p18.clay AND m.best_of = p18.best_of
LEFT JOIN historic_ordered p19
ON m.player1 = p19.player AND m.p1_rn = p19.rn+9 AND m.clay = p19.clay AND m.best_of = p19.best_of
LEFT JOIN historic_ordered p110
ON m.player1 = p110.player AND m.p1_rn = p110.rn+10 AND m.clay = p110.clay AND m.best_of = p110.best_of
-- player 2
LEFT JOIN historic_ordered p21
ON m.player2 = p21.player AND m.p2_rn = p21.rn+1 AND m.clay = p21.clay AND m.best_of = p21.best_of
LEFT JOIN historic_ordered p22
ON m.player2 = p22.player AND m.p2_rn = p22.rn+2 AND m.clay = p22.clay AND m.best_of = p22.best_of
LEFT JOIN historic_ordered p23
ON m.player2 = p23.player AND m.p2_rn = p23.rn+3 AND m.clay = p23.clay AND m.best_of = p23.best_of
LEFT JOIN historic_ordered p24
ON m.player2 = p24.player AND m.p2_rn = p24.rn+4 AND m.clay = p24.clay AND m.best_of = p24.best_of
LEFT JOIN historic_ordered p25
ON m.player2 = p25.player AND m.p2_rn = p25.rn+5 AND m.clay = p25.clay AND m.best_of = p25.best_of
LEFT JOIN historic_ordered p26
ON m.player2 = p26.player AND m.p2_rn = p26.rn+6 AND m.clay = p26.clay AND m.best_of = p26.best_of
LEFT JOIN historic_ordered p27
ON m.player2 = p27.player AND m.p2_rn = p27.rn+7 AND m.clay = p27.clay AND m.best_of = p27.best_of
LEFT JOIN historic_ordered p28
ON m.player2 = p28.player AND m.p2_rn = p28.rn+8 AND m.clay = p28.clay AND m.best_of = p28.best_of
LEFT JOIN historic_ordered p29
ON m.player2 = p29.player AND m.p2_rn = p29.rn+9 AND m.clay = p29.clay AND m.best_of = p29.best_of
LEFT JOIN historic_ordered p210
ON m.player2 = p210.player AND m.p2_rn = p210.rn+10 AND m.clay = p210.clay AND m.best_of = p210.best_of
 LIMIT 100
;











