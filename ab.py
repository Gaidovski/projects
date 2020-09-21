#!/usr/bin/env python
# coding: utf-8

# Был запущен сплит-тест (а/б-тест), направленный на улучшение платежной активности пользователей. Вам дан датасет с транзакциями пользователей до и во время теста в контрольной и тестовых группах
# 
# 1. Какие вы можете сделать выводы? Какая группа показала лучшие результаты?
# 2. Можем ли мы как-то оценить из этих данных равномерность и валидность распределения юзеров по разным группам?
# 3. Если не ограничиваться теми данными, которые приведены в рамках этого задания, что ещё вы бы посчитали для оценки результатов групп?
# 
# **Описание данных:**
# 
# В таблице users_ приведена информация о том, какой юзер в какой момент времени попал в а/б тест:
# - tag - лэйбл группы (control - контрольная, остальные - тестовые)
# - ts - время, когда впервые был выдан tag. То есть, все события до наступления времени ts происходили с юзером до попадания в а/б тест
# - user_uid - внутренний id юзера (для матчинга со второй таблицей)
# - registration_time - время регистрации пользователя в сервисе
# - conv_ts - время совершения первой покупки пользователем в сервисе
# 
# В таблице purchases_ приведена информация о транзакциях пользователей из таблицы users_ до и во время а/б теста:
# - user_uid - внутренний id юзера (для матчинга со второй таблицей)
# - time - время совершения транзакции
# - consumption_mode - вид потребления контента (dto - единица контента куплена навсегда, rent - единица контента взята в аренду, subscription - оформлена подписка)
# - element_uid - уникальный id единицы контента или подписки
# - price - цена (преобразованная)
# 
# Значения в полях price и всех полях, указывающих на время - преобразованы. Это значит, что значение в таблице не настоящее, но является линейным преобразованием реального значения, где ко всем значениям одного поля применено одно и то же преобразование - между ними сохранено отношение порядка. Ко всем полям, обозначающим время, применено одно и то же преобразование.

# # Выводы
# 
# ## 1. Какие вы можете сделать выводы? Какая группа показала лучшие результаты?
# 
# Тестовая группа №3 показала лучшие резульатаы по ARPU
# 
# 69.59 рубля против 68.34 рубля в контрольной
# 
# ## 2. Можем ли мы как-то оценить из этих данных равномерность и валидность распределения юзеров по разным группам?
# 
# Всего 155104 пользователя совершили покупки после деления их на группы. 22% пользователей сервмса от общего числа прин
# Пользователи распределились по группам не совсем равномерно

# In[1]:


# Необходимые библиотеки для исследования 

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as st


# In[2]:


sns.set(
    font_scale=2,
    style="whitegrid",
    rc={'figure.figsize':(16,6)}
        )


# In[3]:


#информация о том, какой юзер в какой момент времени попал в а/б тест
users = pd.read_csv('/Users/gaidovski/Downloads/Архив/users_.csv')
users.head()


# In[4]:


users.isna().sum()


# In[5]:


users.dtypes


# In[6]:


users.shape
print(f"{users.shape[0]} количество строк")


# In[7]:


print(f"{users.user_uid.nunique()} уникальных пользователей")


# **Комментарий**
# 
# 
# Заметим, что количество строк в users больше, чем уникальных пользователей.
# Посмотрим на уникальные id. Не было ли такого, что один пользователь попал в разные группы

# In[8]:


dupl = users.groupby('user_uid', as_index=False).agg({'registration_time':'count'}) .sort_values(by='registration_time', ascending=False).rename(columns={"registration_time": "number_of_tags"})
dupl['more_than_one'] = dupl.number_of_tags >= 2
dupl.head()


# **Комментарий**
# 
# Видим, что некоторым пользователям тэг был присвоен несколько раз.
# Посмотрим, что именно происходило с первым пользователем по количеству присвоенных тэгов

# In[9]:


users.query("user_uid == '6c2289ccfa50dd5b669b159fe12ca4ed'").sort_values(by='ts', ascending=False)


# In[10]:


print(f"{dupl.more_than_one.sum()} пользователям был присвоен тэг больше 1 раза")


# In[13]:


dupl.query("more_than_one == True").number_of_tags.sum()


# In[15]:


users.shape[0] - dupl.query("more_than_one == True").number_of_tags.sum() + dupl.more_than_one.sum()


# In[ ]:





# In[16]:


users.conv_ts.hist(alpha=0.5, color='green', bins=50, label='Первая покупка')
users.registration_time.hist(alpha=0.5, bins=50, label="Регистрация")

#plt.xticks(rotation=90) 
plt.title('Распределение регистраций и первых покупок по времени')
plt.xlabel('Время')
plt.legend();


# **Комментарий**
# 
# Видим, что после условного времени 29000 произошел рост числа регистраций

# In[17]:


#данные о транзакциях пользователей из таблицы users_ до и во время а/б теста:
purchases = pd.read_csv('/Users/gaidovski/Downloads/Архив/purchases_.csv', sep=',')
purchases['price'] = purchases.price.round(1)
purchases.head()


# In[18]:


purchases.shape


# In[19]:


purchases.isna().sum()


# In[20]:


purchases.dtypes


# In[21]:


print(f"{purchases.user_uid.nunique()} уникальных пользователей")


# In[22]:


#зададим отдельную таблицу по количеству транзакций
number_of_purchases = purchases.groupby('user_uid',as_index=False).agg({'time':'count'}) .rename(columns={"time": "number_of_purchases"}) .sort_values(by='number_of_purchases', ascending=False)
#посмотрим на описантельные статистики покупок пользователей
number_of_purchases.describe()


# In[23]:


#посмотрим на распределение по времени
purchases.time.hist(alpha=0.5, color='green', bins=50)

plt.title('Распределение транзакций времени')
plt.xlabel('Время');


# In[24]:


number_of_purchases.number_of_purchases.hist(alpha=0.5, color='blue', bins=100)

plt.title('Распределение числа покупок на пользователя')
plt.xlabel('Количество покупок на одного пользователя');


# **Комментарий**
# 
# - В среднем пользователи совершили 2.2 транзакции за расмотренный условный период 
# - 25% пользователей совершили 3 и более транзакции
# - Максимальное количество покупок на одного юзера - 60
# - Большинство пользователей совершили 1 покупку
# - Распределение покупок по времени имеет явную цикличность (возможно, по дня недели)

# In[25]:


#количество приобретений контента по виду потребления контента
popular_consumption_mode = purchases.pivot_table(index = 'consumption_mode', values = 'user_uid', 
                        aggfunc = 'count', fill_value=0).reset_index() \
.rename(columns={"user_uid": "number_of_transactions"}) \
.sort_values(by='number_of_transactions', ascending=False)
popular_consumption_mode.head()


# In[26]:


purchases.pivot_table(index = 'consumption_mode', values = 'price', 
                        aggfunc = 'mean', fill_value=0).reset_index() \
#.rename(columns={"user_uid": "number_of_transactions"}) \
#.sort_values(by='number_of_transactions', ascending=False)


# In[27]:


#количество транзакций, средняя цена и выручка по каждому виду потребления
consumption_mode = purchases.groupby('consumption_mode', as_index=False) .agg({'time':'count','price':'mean'}) .rename(columns={"time": "number_of_transactions", 'price':'mean_price'})
consumption_mode['revenue'] = consumption_mode.number_of_transactions * consumption_mode.mean_price
consumption_mode = consumption_mode.sort_values(by='revenue', ascending=False)
consumption_mode['share_of_revenue'] = consumption_mode.revenue / consumption_mode.revenue.sum()
consumption_mode['share_of_transactions'] = consumption_mode.number_of_transactions / consumption_mode.number_of_transactions.sum()
round(consumption_mode,2)


# - "Подписка" принесла 45% выручки за расмотренный период 
# - Так же подписка - самый дорогой вид потребления. Средняя цена - 45.8
# - Покупка контента и подписка почти одинаково популярны 37% и 36% соответсвенно
# - Меньше всего клиенты сервиса пользовались "арендой". Так же аренда контента внесла наименьший вклад в выручку сервиса 20%
# 

# In[28]:


#общая таблица
us_pur = purchases.merge(users, on='user_uid')


# In[29]:


#оставим транзакции, которые были совершены после определения тэга для юзера
us_pur_ab = us_pur.query("time > ts")
us_pur_ab.head()


# In[30]:


us_pur_ab.shape


# In[ ]:





# In[ ]:





# In[31]:


sum_per_user = us_pur_ab.groupby(['user_uid', 'tag'], as_index = False) .agg({'price': 'sum'}).sort_values(by='tag')
sum_per_user


# In[43]:


#посчитаем ARPU и распределение пользователей по группам
groups = sum_per_user.groupby(['tag'], as_index = False).agg({'price':'mean','user_uid':'count'}). rename(columns={"price": "ARPU", "user_uid":"number_of_users"})
groups['test_users_share'] = groups.number_of_users / groups.number_of_users.sum()
groups = round(groups, 2)
groups.head()


# **Комментарий**
# 
# - на первый взгляд пользователи по тестовым группам распределились относительно равномерно, но в контрольной группе значения сильно меньше
# - проверим равномерность распределения
# - наибольшее различие по ARPU у контрольной группы с test3
# - сравним ARPU 

# In[33]:


from scipy.stats import chisquare


# In[34]:


chi_pvalue = chisquare(groups.number_of_users)[1]
if 0.05 >= chi_pvalue:
    print('Отклоняем H0, пользователи распредены не равномерно по группам')
else:
    print('Не отклоняем H0, пользователи распредены равномерно по группам')


# In[35]:


test_3 = sum_per_user.query("tag == 'test3'")['price']
control = sum_per_user.query("tag == 'control'")['price']


# In[36]:


#boot control
boot_it = 1000
boot_data_c = []
boot_conf_level = 0.95

for i in range(boot_it):
    samples = control.sample(len(control), replace=True)
    boot_data_c.append(np.mean(samples))

print(f'Original:{np.mean(control)}, Boot: {np.mean(boot_data_c)}')


# In[37]:


#boot test_3
boot_it = 1000
boot_data_3 = []
boot_conf_level = 0.95

for i in range(boot_it):
    samples = test_3.sample(len(test_3), replace=True)
    boot_data_3.append(np.mean(samples))

print(f'Original:{np.mean(test_3)}, Boot: {np.mean(boot_data_3)}')


# In[38]:


left_cl = (1 - boot_conf_level) / 2
right_cl = (1 + boot_conf_level) / 2
cl_3 = pd.Series(boot_data_3).quantile([left_cl,right_cl])
cl_c = pd.Series(boot_data_c).quantile([left_cl,right_cl])


# In[39]:


plt.hist(pd.Series(boot_data_c), bins=50, alpha=0.5, label='control')
plt.hist(pd.Series(boot_data_3), bins=50, alpha=0.5, label='test_3')
plt.style.use('ggplot')
plt.vlines(cl_3,ymin=0,ymax=50,linestyle='--')
plt.vlines(cl_c,ymin=0,ymax=50,linestyle='-')
plt.title('бутстрап test_3 vs control')
plt.legend();


# In[40]:


cl_pvalue = pd.Series(pd.Series(boot_data_3) - pd.Series(boot_data_c)).quantile([left_cl,right_cl])
plt.hist(pd.Series(boot_data_3) - pd.Series(boot_data_c), bins=50, alpha=0.5, color='red')
plt.title('разница бутсрап распределений test_3 vs control')
plt.vlines(cl_pvalue,ymin=0,ymax=50,linestyle='--');


# **Комментарий**
# 
# 0 не попадает в разницу между бутстрап распределениями. Можем считать, что ARPU группы test_3 больше статистически значимо больше, чем у контрольной группы

# In[45]:


155104 / 694819 


# In[ ]:




