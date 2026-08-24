SELECT MAX(daily_total) AS max_daily_total
FROM (
    SELECT SUM(amount) AS daily_total
    FROM orders
    WHERE order_date >= DATE '2024-01-01'
      AND order_date <= DATE '2024-01-31'
    GROUP BY customer_id, order_date
) t;