from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# from sqlalchemy.sql.ddl import CreateTable
# from sqlalchemy.sql.schema import Column, MetaData, Table
# from sqlalchemy import create_engine
# from sqlalchemy import MetaData
# from sqlalchemy.sql.sqltypes import Date, DateTime, Integer
# import contextlib
# import sqlalchemy.exc

"""
# verimiz
data = pd.read_csv('train.csv')

metadata = MetaData()

# Veri tabanının tablosunu oluştur.
# date store item sales
stock_table = Table(
    "stock",
    metadata,
    Column("date", Date, primary_key=True),
    Column("store", Integer, primary_key=True),
    Column("item", Integer, primary_key=True),
    Column("sales", Integer)
)

print(stock_table.primary_key)

# sondaki veri tabanı adı
database_name = 'stock'

with contextlib.suppress(sqlalchemy.exc.ProgrammingError):
    with create_engine(
            "postgresql://postgres:postgres@localhost:5432/", isolation_level='AUTOCOMMIT').connect() as conn:
        conn.execute(f'CREATE DATABASE {database_name}')

engine = create_engine(
    f'postgresql://postgres:postgres@localhost:5432/{database_name}')

# tabloyu oluşturmak için
metadata.create_all(engine)


conn = engine.connect()

# verileri veri tabanına insert etme
for date, store, item, sales in data.values:
    # print(date, store, item, sales)
    trans = conn.begin()
    try:
        conn.execute(
            f"INSERT INTO stock (date,store,item,sales) VALUES ('{date}',{store},{item},{sales})")
    except:
        trans.rollback()
        raise
    else:
        trans.commit()

"""


data = pd.read_csv('source/train.csv')
number_of_store = np.max(data['store'])
print("data is \n", data.head())

# her bir magazanın verilerini tutan hash map
store_dict = {store_id: data.loc[data['store'] == store_id].iloc[:, [
    0, 2, 3]] for store_id in range(1, number_of_store+1)}

# içeriğini görmek için
# print("store sales is\n", store_dict)


store_item_sales_dict = {}

# birinci dükkanda satılan 1. ürünler ...
for store_id in store_dict.keys():
    store_data = store_dict[store_id]
    max_item = np.max(store_data['item'])
    print("maksimum ürün sayısı", max_item)
    for item_id in range(1, max_item+1):
        print(store_id, " dukkanındaki ", item_id, ".ürünü.")
        store_i_store = {item_id: store_data.loc[data['item'] == item_id]}
        print(store_i_store)
        plt.title(
            f"{store_id} dukkanındaki {item_id}.ürünü satış zaman grafiği.")
        # plt.title("ilk üç dükkanın 1.ürününün satış zaman grafiği")
        date = store_i_store[item_id]['date']

        # datetime(int(date[0].split('-')[0]),
        #          int(date[0].split('-')[1]), int(date[0].split('-')[2]))
        
        item_store_tuple =(store_id,item_id)
        date_time_list  = list(map(lambda date: datetime(int(date.split('-')[0]), int(date.split(
            '-')[1]), int(date.split('-')[2])), date))
        sales_list  = store_i_store[item_id]['sales']

        store_item_sales_dict[item_store_tuple] = store_i_store[item_id]

        plt.plot(date_time_list, sales_list)

        plt.savefig(f"{store_id}_{item_id}.png")
        plt.close()

item_sales = []
item_sales.append(None)
for item_id in range(1, max_item+1):
    item_sales.append(store_item_sales_dict[(1, item_id)])
    for store_id in range(2, number_of_store):
        # store_item_sales_dict[(store_id, item_id)].iloc[2,2:] + store_item_sales_dict[(store_id, 3)].iloc[2,2:]
        item_sales[item_id].iloc[:,2:] = np.add( item_sales[item_id].iloc[:,2:], store_item_sales_dict[(store_id, item_id)].iloc[:,2:])
print(item_sales)

# tüm stoklar bazında ürün yazdırma.
for item_id in range(1, max_item+1):
    print("item " , item_id, " tüm dükkan stokları")

    date_time_list  = list(map(lambda date: datetime(int(date.split('-')[0]), int(date.split(
            '-')[1]), int(date.split('-')[2])), date))
    sales_list = item_sales[item_id]['sales']


    plt.plot(date_time_list,sales_list)
    plt.savefig(f"all_stock_{item_id}.png")
    plt.close()


# print(data.head())

# print(np.max(data['store']))
# first_store_sales = data.loc[data['store'] == 1]

# print(first_store_sales.head())

# plt.scatter(pf.iloc[:,0],pf.iloc[:,1])
# plt.show()


