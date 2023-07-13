import psycopg2

# Database connection details
host = "34.116.206.232"
database = "postgres"
user = "postgres"
password = "postgres"

try:
    connection = psycopg2.connect(
        host=host,
        port="5432",
        database=database,
        user=user,
        password=password  
    )
    print("Successfully connected to the PostgreSQL database!")
except (Exception, psycopg2.Error) as error:
    print("Error while connecting to PostgreSQL:", error)

# Create a cursor object to interact with the database
cursor = connection.cursor()

# Example query execution
cursor.execute("SELECT * FROM your_table")
results = cursor.fetchall()

# Process the query results
for row in results:
    print(row)

# Close the cursor and connection
cursor.close()
connection.close()