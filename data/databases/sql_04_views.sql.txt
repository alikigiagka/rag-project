-- View = Query
-- σχήμα της όψης είναι το σχήμα του αιτήματος
-- θα μπορούσαμε να ορίσουμε την όψη ως ιδεατό (virtual) πίνακα


-- q01 δημιουργία όψης
create view twentyfirstcenturycd as
select * 
from cd 
where year > 1999;


-- q02 χρήση όψης
select tf.cid, tf.ctitle, tf.year, count(*)
from twentyfirstcenturycd tf join track t using (cid)
group by tf.cid, tf.ctitle, tf.year
order by tf.ctitle;


-- q03 ιδεατή εκτέλεση του παραπάνω
create temporary table temp as
select * from cd where year > 1999;

select tf.cid, tf.ctitle, tf.year, count(*)
from temp tf join track t using (cid)
group by tf.cid, tf.ctitle, tf.year
order by tf.ctitle;

drop table temp;


-- q04 στην όμως πράξη το DBMS κάνει αυτό (rewritting):
select tf.cid, tf.ctitle, tf.year, count(*)
from (select * from cd where year > 1999) tf join track t using (cid)
group by tf.cid, tf.ctitle, tf.year
order by tf.ctitle;


-- q05 ή για την ακρίβεια αυτό! (αν έχει καλό βελτιστοποιητή)
select tf.cid, tf.ctitle, tf.year, count(*)
from cd tf join track t using (cid)
where year > 1999
group by tf.cid, tf.ctitle, tf.year
order by tf.ctitle;


-- q06 ορισμός όψης πάνω σε όψη
create view twentyfirstcenturyperf as
select name 
from (twentyfirstcenturycd join track using(cid)) join performer using(pid);


-- q07 τί γίνεται αν διαγράψω την όψη;
drop view twentyfirstcenturycd;


-- q08 with statement
-- with t1 (col1, col2) as
--         (select ...),
--      t2 (col1) as
--         (select ...)
-- select ...
-- from cd, t1, t2
-- ...
with twentyfirstcenturycd as
(select * 
 from cd 
 where year > 1999)
select tf.cid, tf.ctitle, tf.year, count(*)
from twentyfirstcenturycd tf join track t using (cid)
group by tf.cid, tf.ctitle, tf.year
order by tf.ctitle;
