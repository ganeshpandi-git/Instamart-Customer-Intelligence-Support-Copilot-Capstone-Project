
-- 1. Check the number of rows in each table
SELECT 'customers' AS table_name, COUNT(*) AS row_count FROM customers
UNION ALL
SELECT 'products', COUNT(*) FROM products
UNION ALL
SELECT 'orders', COUNT(*) FROM orders
UNION ALL
SELECT 'tickets', COUNT(*) FROM tickets;

-- 2. Check the number of orders by status
SELECT status, COUNT(*) AS order_count
FROM orders
GROUP BY status
ORDER BY status;

-- 3. Check the top 10 customers by total spending
SELECT
    customer_id,
    ROUND(SUM(order_amount), 2) AS total_spending
FROM orders
WHERE status = 'completed'
GROUP BY customer_id
ORDER BY total_spending DESC
LIMIT 10;

-- 4. Check the number of tickets by category
SELECT true_category, COUNT(*) AS ticket_count
FROM tickets
GROUP BY true_category
ORDER BY ticket_count DESC;
