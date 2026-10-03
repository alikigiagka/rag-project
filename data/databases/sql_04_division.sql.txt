-- ΟΛΑ ΟΣΑ ΠΡΕΠΕΙ ΝΑ ΞΕΡΕΤΕ ΓΙΑ ΤΗ ΔΙΑΙΡΕΣΗ ΚΑΙ ΔΕΝ ΤΟΛΜΟΥΣΑΤΕ ΝΑ ΡΩΤΗΣΕΤΕ

-- Οι πίνακες για τα παρακάτω παραδείγματα βρίσκονται στη βάση examples

-- 1ο ΠΑΡΑΔΕΙΓΜΑ

-- Έχουμε το σχεσιακό σχήμα:

-- student (am, name)
-- course (km, title)
-- takes (am, km)

-- Θέλουμε όλους τους φοιτητές που πήραν ΟΛΑ τα μαθήματα. 
-- Σε σχεσιακή άλγεβρα θα απαντούσαμε το ερώτημα ως εξής:

-- temp = πkm(course)
-- answer1 = takes / temp
-- και επειδή θέλουμε όλα τα στοιχεία των φοιτητών, η τελική απάντηση είναι:
-- answer2 = student |X| answer1 (όπου |Χ| η φυσική σύζευξη)

-- Στην SQL έχουμε πολλούς τρόπους για να πετύχουμε το ίδιο αποτέλεσμα.


-- Με διπλή άρνηση: οι φοιτητές για τους οποίους ΔΕΝ υπάρχει μάθημα που να ΜΗΝ το έχουν πάρει
select * 
from student s 
where not exists (select * 
                   from course c 
                   where not exists (select * 
                                      from takes t 
                                      where t.km=c.km and t.am=s.am));


-- Με διαφορά συνόλων: οι φοιτητές για τους οποίους αν αφαιρεθεί από το σύνολο όλων των 
-- μαθημάτων που υπάρχουν το σύνολο των μαθημάτων που πήραν οι φοιτητές, προκύπτει το κενό σύνολο
select * 
from student s 
where not exists (
  (select km 
   from course) 
   except 
  (select km 
   from takes t 
   where t.am=s.am));


-- Με μέτρημα: οι φοιτητές για τους οποίους το πλήθος των μαθημάτων που πήραν ισούται 
-- με το πλήθος όλων των μαθημάτων που υπάρχουν
select * 
from student s 
where 
  (select count(*) 
   from takes t 
   where s.am=t.am)
   =
  (select 
   count(*) 
   from course);


-- Με μέτρημα μέσω ομαδοποίησης: ίδια εξήγηση με το αμέσως παραπάνω
select * 
from student s 
where am in (select am 
             from takes
             group by am
             having count(*) = (select count(*) from course));




-- 2ο ΠΑΡΑΔΕΙΓΜΑ

-- Στη διαφάνεια 18 του αρχείου db-06.pdf (Διάλεξη 6) δίνεται ένα παράδειγμα 
-- διαίρεσης στη σχεσιακή άλγεβρα. 

-- με διπλή άρνηση
select distinct p1,p3 
from a 
where not exists
 (select * 
  from b
  where not exists
   (select * 
    from a a2 
    where a2.p2=b.p2 and a2.p4=b.p4 and a2.p1=a.p1 and a2.p3=a.p3));


-- με διαφορά συνόλων
select distinct p1,p3 
from a
where not exists (
 (select p2,p4 from b) 
 except 
 (select p2,p4 from a a2 where a2.p1=a.p1 and a2.p3=a.p3));


-- με μέτρημα
select distinct p1,p3 
from a 
where 
 (select count(*) 
  from a a2 
  where a.p1=a2.p1 and a2.p3=a.p3)
  =
 (select count(*) from b);


-- με μέτρημα μέσω ομαδοποίησης
select distinct p1,p3 
from a 
where (p1, p3) in
 (select p1, p3
  from a
  group by p1 ,p3
  having count(*) = (select count(*) from b));



-- Σας δίνονται οι εντολές DDL για τη δημιουργία των παραπάνω
-- πινάκων σε δική σας βάση δεδομένων

create table student (am integer primary key, name varchar(50));
create table course (km integer primary key, title varchar(50));
create table takes (am integer, km integer, 
  constraint fkam foreign key (am) references student(am), 
  constraint fkkm foreign key (km) references course(km));

insert into student values (1, 'Edgar F. Codd'), (2, 'Peter Chen'), 
 (3, 'Jim Gray'), (4, 'Mike Stonebraker');
insert into course values (101, 'Databases'), (102, 'Web Programming');
insert into takes values (1, 101), (1, 102), (2, 101), (3, 102), (4, 101), (4, 102);



create table a (p1 char, p2 integer, p3 char, p4 integer);
create table b (p2 integer, p4 integer);

insert into a values ('χ', 1, 'ψ', 2), ('κ', 1, 'λ', 2), ('χ', 3, 'ψ', 4),
 ('α', 3, 'β', 4),  ('γ', 1, 'δ', 2),  ('κ', 3, 'λ', 4);
insert into b values (1, 2), (3, 4);
