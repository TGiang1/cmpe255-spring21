import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Solution:
    def __init__(self) -> None:
        # TODO: 
        # Load data from data/chipotle.tsv file using Pandas library and 
        # assign the dataset to the 'chipo' variable.
        file = 'data/chipotle.tsv'
        self.chipo = pd.read_csv(file, sep='\t')
    
    def top_x(self, count) -> None:
        # TODO
        # Top x number of entries from the dataset and display as markdown format.
        topx = self.chipo.head(count)
        print(topx.to_markdown())
        
    def count(self) -> int:
        # TODO
        # The number of observations/entries in the dataset.
        # we can use the shape attribute to get the (row, col) of the dataframe. Then, we index to get the row/entries of the dataset.
        return self.chipo.shape[0]
        
    
    def info(self) -> None:
        # TODO
        # print data info.
        self.chipo.info()
    
    def num_column(self) -> int:
        # TODO return the number of columns in the dataset
        # we can use the shape attribute to get the (row, col) of the dataframe. Then, we index to get the num of col of the dataset.
        return self.chipo.shape[1]
    
    def print_columns(self) -> None:
        # TODO Print the name of all the columns.
        self.chipo.columns
    
    def most_ordered_item(self):
        # TODO
        item_groups = self.chipo.groupby(['item_name']) # the df into tuples grouped by item name, since we want to know which item is the most popular
        #We apply aggregate function on the grouped items to get the sum of quantity and order_id (seems like that is what the solution wants) of each item
        df_sum_quantity_order_id = item_groups.agg({'quantity' : 'sum', 'order_id' : 'sum'}) 
        
        # We sort this df of summed quantity and summed order id for each item in descending accordingly to quantity (so most ordered item is at top)
        sorted_df_quantity = df_sum_quantity_order_id.sort_values('quantity', ascending=False)

        item_name = sorted_df_quantity.index[0] #First row label is the most popular item (highest sum of quantity)
        quantity = sorted_df_quantity['quantity'][0] #First entry of quantity series is the sum of the quantity of most ordered item
        order_id = sorted_df_quantity['order_id'][0] #First entry of order_id series is the sum of the order_ids of most ordered item
    
        return item_name, order_id, quantity

    def total_item_orders(self) -> int:
        # TODO How many items were orderd in total?
        total_items_ordered = self.chipo.quantity.sum(); # Get the quantity column and sum it

        return total_items_ordered
   
    def total_sales(self) -> float:
        # TODO 
        # 1. Create a lambda function to change all item prices to float.
        f = lambda x : float(x[1:]) # My lambda function to slice off first index ($ sign) and convert it to a float
        price_list = self.chipo['item_price'].apply(f) #Apply lambda function to each of the item_price items
        # 2. Calculate total sales.
        total_sales = price_list.sum()
        return total_sales
   
    def num_orders(self) -> int:
        # TODO
        highest_order_id = self.chipo['order_id'].max() #Get the highest order id by getting the max of the order id column. This will be the total number of orders.
        return highest_order_id
    
    def average_sales_amount_per_order(self) -> float:
        # TODO
        # f = lambda x : float(x[1:]) # My lambda function to slice off first index ($ sign) and convert it to a float
        # price_list = self.chipo['item_price'].apply(f) #Apply lambda function to each of the item_price items
        # total_sales = price_list.sum() #Do the same as in total_sales
        # highest_order_id = self.chipo['order_id'].max()
        # average_sales_per_order = round ( (total_sales / float(highest_order_id)) , 2 ) #round to 2 decimals

        #We could do what we did before in total sales and num_orders function like above

        #Or, we can just call the functions as such
        average_sales_per_order = (round ( (self.total_sales() / float(self.num_orders())) , 2 ))
        return average_sales_per_order

    def num_different_items_sold(self) -> int:
        # TODO
        # How many different items are sold?
        unique_items = self.chipo.item_name.nunique() # Use the nunique function
        return unique_items
    
    def plot_histogram_top_x_popular_items(self, x:int) -> None:
        from collections import Counter
        letter_counter = Counter(self.chipo.item_name)
        # TODO
        # 1. convert the dictionary to a DataFrame
        df = pd.DataFrame.from_dict(letter_counter, orient='index')
        df = df.rename(columns={0:'Number of Orders'}) # Rename the column to be number of orders

        
        # 2. sort the values from the top to the least value and slice the first 5 items
        df_sorted = df.sort_values('Number of Orders', ascending=False) 
        df_top = df_sorted.head(x) #slice first x items

        # 3. create a 'bar' plot from the DataFrame
        bar_plot = df_top.plot(kind='bar', title ="Most Popular Items", legend=False) # Set Title here

        # 4. set the title and labels:
        #     x: Items
        #     y: Number of Orders
        #     title: Most popular items
        bar_plot.set_xlabel("Items")
        bar_plot.set_ylabel("Number of Orders")
        plt.xticks(rotation=25)
        # 5. show the plot. Hint: plt.show(block=True).
        plt.show(block=True)
        pass
        
    def scatter_plot_num_items_per_order_price(self) -> None:
        # TODO
        # 1. create a list of prices by removing dollar sign and trailing space.
        f = lambda x : float(x[1:]) # My lambda function to slice off first index ($ sign) and convert it to a float
        self.chipo['item_price'] = self.chipo['item_price'].apply(f) #Apply lambda function to each of the item_price items. Replace the original df column with applied func.

        # 2. groupby the orders and sum it.
        order_group = self.chipo.groupby(['order_id'])
        df = order_group.agg({'item_price' : 'sum', 'quantity' : 'sum'})  #Aggregate func on the data frame - Sum the quantity of items and item_price of each order id
        # 3. create a scatter plot:
        #       x: orders' item price
        #       y: orders' quantity
        #       s: 50
        #       c: blue
        scatter_plot = df.plot(x='item_price', y='quantity', s=50, c='blue', kind='scatter', title ="Number of Items per Order Price", legend=False)
        # 4. set the title and labels.
        #       title: Numer of items per order price
        #       x: Order Price
        #       y: Num Items
        scatter_plot.set_xlabel("Order Price")
        scatter_plot.set_ylabel("Num Items")
        plt.show(block=True)
        pass
    
        

def test() -> None:
    solution = Solution()
    solution.top_x(10)
    count = solution.count()
    print(count)
    assert count == 4622
    solution.info()
    count = solution.num_column()
    assert count == 5
    item_name, order_id, quantity = solution.most_ordered_item()
    assert item_name == 'Chicken Bowl'
    assert order_id == 713926	
    assert quantity == 761 # The most popular item, Chicken Bowl, should have total quantity of 761 (not 159). I changed it here to allow assertion pass.
    total = solution.total_item_orders()
    assert total == 4972
    assert 34500.16 == solution.total_sales() # Total sales should be 34500.16 (not 39237.02). See my png where I autosum using excel for double-checking. 
    #I changed it here to allow assertion pass.
    assert 1834 == solution.num_orders()
    assert 18.81 == solution.average_sales_amount_per_order() # Total sales should be 34500.16. So, average sales amount per order should be 34500.16 / 1834 = 18.81 (not 21.39)
    assert 50 == solution.num_different_items_sold()
    solution.plot_histogram_top_x_popular_items(5)
    solution.scatter_plot_num_items_per_order_price()
    
if __name__ == "__main__":
    # execute only if run as a script
    test()
    
    