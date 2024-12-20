def calculate_my_price(my_cur_price, my_cur_num, trade_price, trade_num):
    if my_cur_num + trade_num < 100:
        print('my_cur_num={} trade_num={} u have no enough num to handle'.format(my_cur_num, trade_num))
        return
    spend_money = round(my_cur_price * my_cur_num + trade_price * trade_num, 3)
    my_new_num = my_cur_num + trade_num
    my_new_price = round(spend_money / my_new_num, 3)
    action = 'sold' if trade_num < 0 else 'bought'
    print('I {} {} at the price {}. and after this trade:'.format(action, abs(trade_num), trade_price))
    print(f"{'spend_money':>12} {'my_new_price':>12} {'my_new_num':>12}  ")
    print("%12s %12s %12s" % (spend_money, my_new_price, my_new_num))
    return my_new_price, my_new_num


if __name__ == '__main__':
    my_cur_price, my_cur_num = 20.79, 1100
    my_trade_list = [(21.41, -300), (20.17, 300)]
    for one_trade in my_trade_list:
        (trade_price, trade_num) = one_trade
        my_cur_price, my_cur_num = calculate_my_price(my_cur_price, my_cur_num, trade_price, trade_num)
