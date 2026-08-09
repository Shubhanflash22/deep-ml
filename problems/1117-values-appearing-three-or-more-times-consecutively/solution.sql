SELECT DISTINCT num
FROM (
    SELECT
        num,
        LAG(num, 1) OVER (ORDER BY id) AS prev1,
        LEAD(num, 1) OVER (ORDER BY id) AS next1,
        LEAD(num, 2) OVER (ORDER BY id) AS next2
    FROM Logs
) t
WHERE num = prev1
   OR (num = next1 AND num = next2)
ORDER BY num;