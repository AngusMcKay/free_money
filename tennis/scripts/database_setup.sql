CREATE DATABASE tennis;

USE tennis;

DROP TABLE IF EXISTS mens;
CREATE TABLE mens (
	atp int,
    best_of int,
    comment nvarchar(64),
    date date, 
    location nvarchar(64),
    winner nvarchar(64),
    round nvarchar(64),
    carpet int,
    clay int,
    grass int,
    hard int,
    atp250 int,
    atp500 int,
    grand_slam int,
    international int,
    international_gold int,
    masters int,
    masters_1000 int,
    masters_cup int,
    player1  nvarchar(64),
    p1Rank int,
    p1Pts int,
    p1sets int,
    p1games1 int,
    p1games2 int,
    p1games3 int,
    p1games4 int,
    p1games5 int,
    b365P1 float,
    b365P1prob float,
    player2 nvarchar(64),
    p2Rank int,
    p2Pts int,
    p2sets int,
    p2games1 int,
    p2games2 int,
    p2games3 int,
    p2games4 int,
    p2games5 int,
    b365P2 float,
    b365P2prob float,
    b365bookiesgain float,
    player1Wins int,
    
    INDEX(player1(32)),
    INDEX(player2(32))
);


LOAD DATA LOCAL INFILE '/home/angus/projects/betting/tennis/all_data/mens_data_reformatted.csv' 
INTO TABLE mens 
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

SELECT *
FROM mens
LIMIT 1000;



DROP TABLE IF EXISTS men_players_historic;
CREATE TABLE men_players_historic (
	atp int,
    best_of int,
    comment nvarchar(64),
    date date, 
    location nvarchar(64),
    winner nvarchar(64),
    round nvarchar(64),
    carpet int,
    clay int,
    grass int,
    hard int,
    atp250 int,
    atp500 int,
    grand_slam int,
    international int,
    international_gold int,
    masters int,
    masters_1000 int,
    masters_cup int,
    player  nvarchar(64),
    playerRank int,
    playerPts int,
    playersets int,
    playergames1 int,
    playergames2 int,
    playergames3 int,
    playergames4 int,
    playergames5 int,
    b365player float,
    b365playerprob float,
    opponent nvarchar(64),
    opponentRank int,
    opponentPts int,
    opponentsets int,
    opponentgames1 int,
    opponentgames2 int,
    opponentgames3 int,
    opponentgames4 int,
    opponentgames5 int,
    b365opponent float,
    b365opponentprob float,
    b365bookiesgain float,
    playerWins int,
    
    INDEX(player),
    INDEX(opponent)
);


INSERT INTO men_players_historic
SELECT * FROM mens;

INSERT INTO men_players_historic
SELECT atp,best_of,comment,date,location,winner,round,carpet,clay,grass,hard,
		atp250,atp500,grand_slam,international,international_gold,masters,masters_1000,masters_cup,
        player2,p2Rank,p2Pts,p2sets,p2games1,p2games2,p2games3,p2games4,p2games5,b365P2,b365P2prob,
        player1,p1Rank,p1Pts,p1sets,p1games1,p1games2,p1games3,p1games4,p1games5,b365P1,b365P1prob,
        b365bookiesgain,(1-player1Wins) as playerWins
FROM mens;




