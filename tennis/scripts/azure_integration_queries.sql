USE tennis;
SELECT COUNT(1) FROM men_players_historic LIMIT 100;
SELECT COUNT(1) FROM mens LIMIT 100;

select user, host from mysql.user;

# allow root access from non-local
CREATE USER 'root'@'%' IDENTIFIED BY 'root';
GRANT ALL PRIVILEGES ON *.* TO 'root'@'%' WITH GRANT OPTION;
FLUSH PRIVILEGES;

# checking users will show that authentification is different for new root profile
USE mysql;
SELECT * FROM user WHERE user='root';

