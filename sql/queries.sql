-- Q1: WHERE then HAVING
SELECT customer_id, SUM(order_amount) as total_spend
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING total_spend > 8000;

-- Q2: Three-table join (Revenue by city and product category)
SELECT c.city, p.category, SUM(o.order_amount) as revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category;

-- Q3: Scalar subquery (Products priced above average catalogue price)
SELECT product_id, product_name, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products);

-- Q4: LEFT JOIN (Customers who have no completed order)
SELECT c.customer_id
FROM customers c
LEFT JOIN (SELECT DISTINCT customer_id FROM orders WHERE status = 'completed') o 
  ON c.customer_id = o.customer_id
WHERE o.customer_id IS NULL;

-- Q5: Common Table Expression (Spend above average)
WITH customer_spend AS (
    SELECT customer_id, SUM(order_amount) as total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
avg_spend AS (
    SELECT AVG(total_spend) as avg_val FROM customer_spend
)
SELECT COUNT(*) as count_above_avg
FROM customer_spend, avg_spend
WHERE customer_spend.total_spend > avg_spend.avg_val;

-- Q6: Ranking window functions
WITH customer_city_spend AS (
    SELECT c.customer_id, c.city, SUM(o.order_amount) as total_spend
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
)
SELECT customer_id, city, total_spend,
       RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) as rnk,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) as dense_rnk,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) as row_num
FROM customer_city_spend;

-- Q7: Running total (Daily completed revenue)
SELECT SUBSTR(order_ts, 1, 10) as day,
       SUM(order_amount) as daily_revenue,
       SUM(SUM(order_amount)) OVER (ORDER BY SUBSTR(order_ts, 1, 10)) as cumulative_revenue
FROM orders
WHERE status = 'completed'
GROUP BY day
ORDER BY day;