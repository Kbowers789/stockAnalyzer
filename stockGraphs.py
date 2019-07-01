# Katherine Bowers      3/5/2018
# CSCI 6647 Advanced Python for Data Science
# Spring 2018
# Assignment 2 - Using Graphs

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits import mplot3d

# function to calculate the simple moving average crossover indicators
def simpleavg():
    # the initial averages for the first day of 2017 (used to decide action for next day)
    prev50 = np.mean(stockClose[(start-50):start])
    prev200 = np.mean(stockClose[(start-200):start])
    savg50.append(prev50)
    savg200.append(prev200)

    for i in range((start+1), dates.size):
        # calculating current day averages
        avg50 = np.mean(stockClose[(i - 50):i])
        avg200 = np.mean(stockClose[(i - 200):i])

        # comparing previous day's previous averages (prev50 and prev200)
        # and current day's previous averages (avg50 and avg200) in order to
        # determine action code for the current day
        if prev50 < prev200 and avg200 < avg50:
            sac_action.append('b')
            sacBuy.append(avg50)
            sacBuyDates.append(i-start)
        elif prev50 > prev200 and avg200 > avg50:
            sac_action.append('s')
            sacSell.append(avg50)
            sacSellDates.append(i-start)
        else:
            sac_action.append('h')
        # adding current averages to their lists
        savg50.append(avg50)
        savg200.append(avg200)

        # checking against last date in array
        if i == (dates.size-1):
            break
        else:
            # reassigning averages for next day's assessment
            prev50 = avg50
            prev200 = avg200


# function to calculate the moving average crossover/divergence indicators
def macdavg():
    # the initial simple averages for the day 1 of 2017 (used to decide action for day 2)
    prev_ema12 = np.mean(stockClose[(start-12):start])
    prev_ema26 = np.mean(stockClose[(start-26):start])
    prev_macd = prev_ema12 - prev_ema26
    prev_sig = 0
    sema12.append(prev_ema12)
    sema26.append(prev_ema26)
    smacd.append(prev_macd)
    s_sig.append(prev_sig)

    for i in range(0,9):
        k = start - i
        temp_macd = np.mean(stockClose[(k-9):k]) - np.mean(stockClose[(k-9):k])
        prev_sig += temp_macd
    prev_sig = prev_sig/9

    # loop to fill out rest of action list
    for i in range((start+1), dates.size):
        # calculating current day values
        ema12 = ((2 * stockClose[i - 1])/13) + ((11 * prev_ema12)/13)
        ema26 = ((2 * stockClose[i - 1])/27) + ((25 * prev_ema26)/27)
        macd = ema12 - ema26
        sig = ((2 * macd)/10) + ((8 * prev_sig)/10)
        # comparing previous day's previous macd and signal values
        # and current day's previous macd and signal values in order to
        # determine action code for the current day
        if prev_macd < prev_sig and sig < macd:
            macd_action.append('b')
            macdBuy.append(sig)
            macdBuyDates.append(i - start)
        elif prev_macd > prev_sig and sig > macd:
            macd_action.append('s')
            macdSell.append(sig)
            macdSellDates.append(i - start)
        else:
            macd_action.append('h')
        # adding current averages to their lists
        sema12.append(ema12)
        sema26.append(ema26)
        smacd.append(macd)
        s_sig.append(sig)

        # checking against last date in array
        if i == (dates.size-1):
            break
        else:
            # reassigning values for next day's assessment
            prev_ema12 = ema12
            prev_ema26 = ema26
            prev_macd = macd
            prev_sig = sig


# function to calculate relative strength index indicators
def rsicalc():
    # loop to fill out action list
    for i in range((start+1), dates.size):
        gain_cnt = 0
        gain_sum = 0
        loss_cnt = 0
        loss_sum = 0

        # calculating previous day's rsi value
        for j in range(0, 14):
            # k calculated using - 2 because we need to look at
            # the previous day's previous 14 gains or losses to
            # determine current day's action (rather than tomorrow's action)
            k = i - 2 - j
            if stockClose[k - 1] > stockOpen[k - 1]:
                gain_cnt += 1
                gain_sum += stockClose[k - 1] - stockOpen[k - 1]
            else:
                loss_cnt += 1
                loss_sum += stockOpen[k - 1] - stockClose[k - 1]

        gains.append(gain_sum)
        losses.append(0 - loss_sum)

        if loss_cnt == 0:
            rsn = 100
        else:
            rsn = (gain_sum / gain_cnt) / (loss_sum / loss_cnt)

        rsi_val = 100 - (100 / (1 + rsn))
        rsi_vals.append(rsi_val)

        # using rsi value from previous day
        # in order to determine action code for current day
        if rsi_val < 30:
            rsi_action.append('b')
            rsiBuy.append(rsi_val)
            rsiBuyDates.append(i-start)
        elif rsi_val > 70:
            rsi_action.append('s')
            rsiSell.append(rsi_val)
            rsiSellDates.append(i-start)
        else:
            rsi_action.append('h')


# function to calculate on balance volume indicators
def obvcalc():

    # creating slope analysis for initial previous day's previous day
    prev_res = [0, 0]
    x = np.arange(0,20,1)
    obv = 0
    for i in range(start, dates.size):
        # evaluating obv for current day
        if stockClose[i] > stockClose[i - 1]:
            obv += stockVolume[i]
        if stockClose[i] < stockClose[i - 1]:
            obv -= stockVolume[i]
        obvVals.append(obv)

        # initializing an array of all zeros for 20 day calculation
        obv_list = [0] * 20

        # filling out obv_list with relevant values
        d = i - 21  # iterator for dates array within loop
        for j in range(0, 20):
            if stockClose[d] > stockClose[d - 1]:
                obv_list[j] += stockVolume[d]
            elif stockClose[d] < stockClose[d - 1]:
                obv_list[j] -= stockVolume[d]
            else:
                obv_list[j] = 0
            d += 1

        res = np.polyfit(x, obv_list, 1)
        obvSlopes.append(res[0])

        # establishing initial previous day's previous day slope without adding to action list
        if i == start:
            prev_res = res
            continue

        # comparing previous day's previous slope to
        # previous day's slope for action determination
        if prev_res[0] > 0 > res[0]:
            obv_action.append('b')
            obvBuy.append(res[0])
            obvBuyDates.append(i - start)
        elif prev_res[0] < 0 < res[0]:
            obv_action.append('s')
            obvSell.append(res[0])
            obvSellDates.append(i - start)
        else:
            obv_action.append('h')

        # checking against last date in array
        if i == (dates.size-1):
            break
        else:
            # reassigning values for next day's assessment
            prev_res = res


# function to fill out action list based on comparing previous action lists
def combo():
    for i in range(0, len(sac_action)):
        b_count = 0
        s_count = 0
        if sac_action[i] == 'b':
            b_count += 1
        if sac_action[i] == 's':
            s_count += 1
        if macd_action[i] == 'b':
            b_count += 1
        if macd_action[i] == 's':
            s_count += 1
        if rsi_action[i] == 'b':
            b_count += 1
        if rsi_action[i] == 's':
            s_count += 1
        if obv_action[i] == 'b':
            b_count += 1
        if obv_action[i] == 's':
            s_count += 1
        if b_count >= 2:
            combo_action.append('b')
        elif s_count >= 2:
            combo_action.append('s')
        else:
            combo_action.append('h')


# function to run and print simulation based on the action list created for each type of comparison
def print_run(method, action):
    # running simulation
    shares = 0
    transaction = 0
    result = 0
    bank = 1000
    method.append(bank)


    for x in range(start, dates.size):
        y = x - start

        if action[y] == 'b':
            newshares = int(bank / stockOpen[x])
            if newshares == 0:
                print(dates[x], "\tNot enough in bank to buy more shares...")
                continue
            shares += newshares
            transaction = stockOpen[x] * newshares
            bank -= transaction
            method.append(bank)
            print(dates[x], "\tBought ", newshares, " shares for $", transaction, "    \tBank at: $", bank, sep="")
        elif action[y] == 's':
            if shares == 0:
                print(dates[x], "\tNo shares to sell...")
                continue
            transaction = stockOpen[x] * shares
            bank += transaction
            method.append(bank)
            print(dates[x], "\tSold ", shares, " shares for $", transaction, "    \tBank at: $", bank, sep="")
            shares = 0
        else:
            method.append(bank)
            continue

    if shares > 0:
        transaction = stockClose[-1] * shares
        bank += transaction
        method.append(bank)
        print("\nEnd of year:\tSold final ", shares, " shares for $", transaction, sep="")
        print("Bank ended at: $", bank, sep="")
    else:
        print("\nEnd of year:\tNo shares to sell.")
        print("Bank ended at: $", bank, sep="")
    if bank < 1000.00:
        result = 1000.00 - bank
        print("Result was a loss of $", result, sep="")
    if bank > 1000.00:
        result = bank - 1000.00
        print("Result was a gain of $", result, sep="")
    return bank


# Main Program
print("---- Stock Exchange Simulation for 2017 ----\n")
dates = np.loadtxt("Dates.txt", dtype='U')
stockClose = np.loadtxt("AMZNclose.txt")
stockOpen = np.loadtxt("AMZNopen.txt")
stockVolume = np.loadtxt("AMZNvolume.txt")

start = 0
for index in range(0, dates.size):
    if dates[index] == "1/3/2017":
        start = index
        break
# start is now the index of the first day of 2017, and thus the same index for
# the stock closing prices on that same day, and can be used for indicator calculations
# the action list will indicate whether to buy ('b'), sell ('s') or hold ('h') on each day

# initializing action lists with initial buy on first day of 2017, and supporting lists for each method
sac_action = ['b']
savg50 = []
savg200 = []
sacBuyDates = []
sacBuy = []
sacSellDates = []
sacSell = []
sac_run = []

sema12 = []
sema26 = []
smacd = []
s_sig = []
macd_action = ['b']
macdBuyDates = []
macdBuy = []
macdSellDates = []
macdSell = []
macd_run = []

gains = []
losses = []
rsi_vals = []
rsiBuyDates = []
rsiBuy = []
rsiSellDates = []
rsiSell = []
rsi_action = ['b']
rsi_run = []

vol = np.array(stockVolume[start:])
obvVals = []
obvSlopes = []
obv_action = ['b']
obvSell = []
obvSellDates = []
obvBuy = []
obvBuyDates = []
obv_run = []

combo_action = ['b']
combo_run = []

# prepping x axis ticks for all graphs
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthTicks = [0,0,0,0,0,0,0,0,0,0,0,0]

for x in range(1,13):
    diff = 0
    temp = str(x) + '/'
    for num in range(start, dates.size):
        if dates[num-1].find(temp, 0, 3) == -1 and dates[num].find(temp, 0, 3) != -1:
            monthTicks[x-1] = num - start
            num += diff
            break
        else:
            num += 1
            diff += 1

# call to fill out action and average lists for simple moving average crossover method
simpleavg()
savg50 = np.array(savg50)
savg200 = np.array(savg200)

# plotting simple average graphs
plt.figure(1)
plt.title("Simple Average Crossover Analysis for 2017 Closing Prices")
plt.xlabel("Day of Year")
plt.ylabel("Price (in USD)")
plt.xticks(monthTicks, months, rotation=40)

# calculating the curves of the 50-day and 200-day averages over the year
sac50deg = np.polyfit(range(0, stockClose.size - start), savg50, 15)
line50 = np.polyval(sac50deg, range(0, stockClose.size - start))
sac200deg = np.polyfit(range(0, stockClose.size - start), savg200, 15)
line200 = np.polyval(sac200deg, range(0, stockClose.size - start))

# plot of closing prices
plt.scatter(range(0, (stockClose.size - start)), stockClose[start:], color = 'red', s = 4, label = "Closing Price")
# plots of averages
plt.plot(range(0, stockClose.size - start), line50, color = 'green', label = "50-Day Average")
plt.plot(range(0, stockClose.size - start), line200, color = 'blue', label = "200-Day Average")
# plots of action dates
plt.plot(sacBuyDates,sacBuy,marker='o',markersize=15,markeredgecolor='yellow',markeredgewidth=2,markerfacecolor='none',label="Date to Buy",linestyle='None')
plt.plot(sacSellDates,sacSell,marker='o',markersize=15,markeredgecolor='orange',markeredgewidth=2,markerfacecolor='none',label="Date to Sell",linestyle='None')
plt.legend()

# creating an annotation for every action point on graph
for i in range(0, len(sacBuyDates)):
    plt.annotate(dates[sacBuyDates[i]+start], xy=(sacBuyDates[i],sacBuy[i]), xytext=(0,20), textcoords="offset points",arrowprops=dict(arrowstyle="->"))

for i in range(0, len(sacSellDates)):
    plt.annotate(dates[sacSellDates[i]+start], xy=(sacSellDates[i],sacSell[i]), xytext=(0,20), textcoords="offset points",arrowprops=dict(arrowstyle="->"))

# printing run results for SMA Method
print("Method 1 -- Simple Moving Average Crossover Simulation:\n")
sac_bank = print_run(sac_run, sac_action)

# call to fill out action list for moving average crossover/divergence method
macdavg()
sema12 = np.array(sema12)
sema26 = np.array(sema26)
smacd = np.array(smacd)
s_sig = np.array(s_sig)

# plotting moving average crossover/divergence graphs
plt.figure(2)
plt.subplot(3, 1, 1)
plt.title("12-Day and 26-Day Averages for 2017 Closing Prices")
plt.xlabel("Day of Year")
plt.ylabel("Price (in USD)")
plt.xticks(monthTicks, months, rotation=40)

# calculating the curves of the 12-day average, 26-day average, macd, and signal values over the year
sema12deg = np.polyfit(range(0, stockClose.size - start), sema12, 15)
sema12line = np.polyval(sema12deg, range(0, stockClose.size - start))
sema26deg = np.polyfit(range(0, stockClose.size - start), sema26, 15)
sema26line = np.polyval(sema26deg, range(0, stockClose.size - start))


# plot of closing prices and averages
plt.scatter(range(0, (stockClose.size - start)), stockClose[start:], color = 'red', s = 4, label = "Closing Price")
plt.plot(range(0, stockClose.size - start), sema12line, color = 'orange', label = "12-Day Moving Average")
plt.plot(range(0, stockClose.size - start), sema26line, color = 'yellow', label = "26-Day Moving Average")
plt.legend()

# plots of macd and signal values, plus buy/sell action indicators
plt.subplot(3,1,3)
plt.title("Moving Average Crossover Values for 2017 Closing Prices")
plt.xlabel("Day of Year")
plt.ylabel("MACD Score")
plt.xticks(monthTicks, months, rotation=40)

plt.plot(range(0, stockClose.size - start), smacd, color = 'purple', label = "MACD Value")
plt.plot(range(0, stockClose.size - start), s_sig, color = 'pink', label = "Signal Value")

plt.plot(macdBuyDates,macdBuy,marker='+',markersize=10,markeredgecolor='green',markeredgewidth=1.5,markerfacecolor='none',label="Date to Buy",linestyle='None')
plt.plot(macdSellDates,macdSell,marker='x',markersize=7,markeredgecolor='red',markeredgewidth=1.5,markerfacecolor='none',label="Date to Sell",linestyle='None')

# creating an annotation for action points on graph
for i in range(0, len(macdBuyDates)):
    # 'dispersing' annotations to avoid text overlap
    if i == 0:
        os = 'd'
    if os == 'd':
        plt.annotate(dates[macdBuyDates[i]+start], xy=(macdBuyDates[i],macdBuy[i]), xytext=(-60,-20), textcoords="offset points",arrowprops=dict(arrowstyle="->"))
        os = 'u'
    else:
        plt.annotate(dates[macdBuyDates[i]+start], xy=(macdBuyDates[i],macdBuy[i]), xytext=(-20,-40), textcoords="offset points",arrowprops=dict(arrowstyle="->"))
        os = 'd'

for i in range(0, len(macdSellDates)):
    # 'dispersing' annotations to avoid text overlap
    if i == 0:
        os = 'd'
    if os == 'd':
        plt.annotate(dates[macdSellDates[i]+start], xy=(macdSellDates[i],macdSell[i]), xytext=(-60,20), textcoords="offset points",arrowprops=dict(arrowstyle="->"))
        os = 'u'
    else:
        plt.annotate(dates[macdSellDates[i]+start], xy=(macdSellDates[i],macdSell[i]), xytext=(10,40), textcoords="offset points",arrowprops=dict(arrowstyle="->"))
        os = 'd'

plt.legend()

# printing run results for MACD Method
print("\n---------------------\nMethod 2 -- Moving Average Crossover/Divergence:\n")
macd_bank = print_run(macd_run, macd_action)


# call to fill out action list for relative strength index method
rsicalc()
gains = np.array(gains)
losses = np.array(losses)
rsi_vals = np.array(rsi_vals)

# graphing results
plt.figure(3)
plt.subplot(2,1,1)
plt.title("Daily Gain/Loss Values for 2017 Closing Prices")
plt.xlabel("Day of Year")
plt.ylabel("Price (in USD)")
plt.xticks(monthTicks, months, rotation=40)

# creating a mid-point line on x-axis at 0, based on this solution:
# https://stackoverflow.com/questions/18176674/add-an-x-axis-at-0-to-a-pyplot-histogram-with-negative-bars
plt.axhline(0, color='black', lw=1)

plt.bar(range(0, gains.size), gains, width=0.5, color='green', label="Gain")
plt.bar(range(0, losses.size), losses, width=0.5, color='red', label="Loss")
plt.legend()

# plotting rsi values with action points
plt.subplot(2,1,2)
plt.title("RSI Daily Values for 2017 Closing Prices")
plt.xlabel("Day of Year")
plt.ylabel("RSI Value")
plt.xticks(monthTicks, months, rotation=40)

plt.axhline(70, color='red', lw=2)
plt.axhline(30, color='green', lw=2)

plt.bar(range(0, rsi_vals.size), rsi_vals, width=0.5, color='blue', label='RSI Value')
plt.plot(rsiBuyDates,rsiBuy,marker='v',markersize=10,markeredgecolor='orange',markeredgewidth=1.5,markerfacecolor='none',label="Date to Buy",linestyle='None')
plt.plot(rsiSellDates,rsiSell,marker='v',markersize=10,markeredgecolor='yellow',markeredgewidth=1.5,markerfacecolor='none',label="Date to Sell",linestyle='None')
plt.legend()

# creating an annotation for every action point on graph
for i in range(0, len(rsiBuyDates)):
    # 'dispersing' annotations to avoid text overlap
    if i == 0:
        os = 'l'
    if os == 'l':
        plt.annotate(dates[rsiBuyDates[i]+start], xy=(rsiBuyDates[i],rsiBuy[i]), xytext=(-50,75), textcoords="offset points",arrowprops=dict(arrowstyle="->"))
        os = 'c'
    elif os == 'c':
        plt.annotate(dates[rsiBuyDates[i]+start], xy=(rsiBuyDates[i],rsiBuy[i]), xytext=(-15,100), textcoords="offset points",arrowprops=dict(arrowstyle="->"))
        os = 'r'
    else:
        plt.annotate(dates[rsiBuyDates[i]+start], xy=(rsiBuyDates[i],rsiBuy[i]), xytext=(20,50), textcoords="offset points",arrowprops=dict(arrowstyle="->"))
        os = 'l'


for i in range(0, len(rsiSellDates)):
    # 'dispersing' annotations to avoid text overlap
    if i == 0:
        os = 'l'
    if os == 'l':
        plt.annotate(dates[rsiSellDates[i]+start], xy=(rsiSellDates[i],rsiSell[i]), xytext=(-50,20), textcoords="offset points",arrowprops=dict(arrowstyle="->"))
        os = 'c'
    elif os == 'c':
        plt.annotate(dates[rsiSellDates[i]+start], xy=(rsiSellDates[i],rsiSell[i]), xytext=(-15,10), textcoords="offset points",arrowprops=dict(arrowstyle="->"))
        os = 'r'
    else:
        plt.annotate(dates[rsiSellDates[i]+start], xy=(rsiSellDates[i],rsiSell[i]), xytext=(20,30), textcoords="offset points",arrowprops=dict(arrowstyle="->"))
        os = 'l'

# printing run results for RSI method
print("\n---------------------\nMethod 3 -- Relative Strength Index:\n")
rsi_bank = print_run(rsi_run, rsi_action)

# call to fill out action list for on balance volume method
obvcalc()

obvVals = np.array(obvVals)
obvSlopes = np.array(obvSlopes)
volTicks = []
n = -75000000
while n < 80000000:
    volTicks.append(n)
    n += 5000000

volNums = [-75, -70, -65, -60, -55, -50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
# plotting results
plt.figure(4)
plt.title("Stock Volume and OBV Values for 2017 Closing Prices")
plt.xlabel("Day of Year")
plt.ylabel("Volume (in millions of shares)")
plt.xticks(monthTicks, months, rotation=40)
plt.yticks(volTicks, volNums)
plt.axhline(0, color='black', lw=1)

plt.bar(range(0, vol.size), vol, width=1, color='green', label='Stock Volume')
plt.bar(range(0, obvVals.size), obvVals, width=0.65, color='purple', alpha=0.5, label='OBV Value')
plt.legend()

# creating obv slope graph and accompanying animation
afig = plt.figure(5)
plt.title("20-day OBV averages for 2017 Closing Prices")
plt.xlabel("Day of Year")
plt.ylabel("Average Volume (in millions of shares)")
plt.xticks(monthTicks, months, rotation=40)
plt.axhline(0, color='black', lw=1)

obvdeg = np.polyfit(range(0, obvSlopes.size), obvSlopes, 15)
obvline = np.polyval(obvdeg, range(0, obvSlopes.size))

plt.plot(range(0,obvSlopes.size), obvSlopes, color='blue', label='change in 20-day OBV average')

# adding action points with no dates this time
plt.plot(obvBuyDates,obvBuy,marker='+',markersize=15,markeredgecolor='green',markeredgewidth=1.5,markerfacecolor='none',label="Date to Buy",linestyle='None')
plt.plot(obvSellDates,obvSell,marker='x',markersize=10,markeredgecolor='red',markeredgewidth=1.5,markerfacecolor='none',label="Date to Sell",linestyle='None')
plt.legend()

# animation set up
slopeDer = np.polyder(obvline)
slopeDerVal = np.polyval(slopeDer, obvSlopes)
b = obvline - obvSlopes * slopeDerVal

xline = np.linspace(obvSlopes[0], obvSlopes[0] + 2, 50)
yline = slopeDerVal[0] * xline + b[0]
tanLine, = plt.plot(xline, yline, color='orange')


# functions for animation
def init_line():
    pass


def update(frame):
    if obvSlopes[frame] - 2 < 0:
        xline = np.linspace(0, obvSlopes[frame] + 2, 50)
    else:
        xline = np.linspace(obvSlopes[frame] - 2, obvSlopes[frame] + 2, 50)
    yline = slopeDerVal[frame] * xline + b[frame]
    tanLine.set_data(xline, yline)
    return tanLine


ani = anim.FuncAnimation(afig, update, frames=obvSlopes.size, init_func=init_line, interval=100, repeat=False)

# printing results for OBV method
print("\n---------------------\nMethod 4 -- On Balance Volume:\n")
obv_bank = print_run(obv_run, obv_action)

# call to fill out action list based on previous 4 action results
combo()
print("\n---------------------\nMethod 5 -- When two or more methods agree:\n")
combo_bank = print_run(combo_run, combo_action)

# final comparison of methods, based on final bank accounts
print("\n---------------------\nFinal Account Analysis:")
print("Method 1:\t$", sac_bank, sep='')
print("Method 2:\t$", macd_bank, sep='')
print("Method 3:\t$", rsi_bank, sep='')
print("Method 4:\t$", obv_bank, sep='')
print("Method 5:\t$", combo_bank, sep='')

# graphing results of simulations
sac_run = np.array(sac_run)
macd_run = np.array(macd_run)
rsi_run = np.array(rsi_run)
obv_run = np.array(obv_run)
combo_run = np.array(combo_run)

plt.figure(6)
rows = [0, 1, 2, 3, 4]
ax = plt.axes(projection='3d')
plt.title("Stock Buying/Selling Simulation for all Methods")
ax.set_xlabel("Day of Year")
ax.set_ylabel("Method")
ax.set_zlabel('Account Balance (in USD)')
ax.set_yticklabels(['SAC', 'MACD', 'RSI', 'OBV', 'Combo'])
ax.set_yticks(rows)
ax.set_xticks(monthTicks)
ax.set_xticklabels(months)

ax.bar(range(0, sac_run.size), sac_run, zs=rows[0], zdir='y', color='orange', alpha=0.8)
ax.bar(range(0, macd_run.size), macd_run, zs=rows[1], zdir='y', color='pink', alpha=0.8)
ax.bar(range(0, rsi_run.size), rsi_run,  zs=rows[2],zdir='y', color='yellow', alpha=0.8)
ax.bar(range(0, obv_run.size), obv_run,  zs=rows[3],zdir='y', color='purple', alpha=0.8)
ax.bar(range(0, combo_run.size), combo_run,  zs=rows[4],zdir='y', color='cyan', alpha=0.8)

# Method comparison result statements
if sac_bank > macd_bank and sac_bank > rsi_bank and sac_bank > obv_bank and sac_bank > combo_bank:
    print("\nConclusion:\tSimple Moving Average Crossover was the most successful method.")
elif macd_bank > sac_bank and macd_bank > rsi_bank and macd_bank > obv_bank and macd_bank > combo_bank:
    print("\nConclusion:\tMoving Average Crossover/Divergence was the most successful method.")
elif rsi_bank > sac_bank and rsi_bank > macd_bank and rsi_bank > obv_bank and rsi_bank > combo_bank:
    print("\nConclusion:\tRelative Strength Index was the most successful method.")
elif obv_bank > sac_bank and obv_bank > macd_bank and obv_bank > rsi_bank and obv_bank > combo_bank:
    print("\nConclusion:\tOn Balance Volume was the most successful method.")
elif combo_bank > sac_bank and combo_bank > macd_bank and combo_bank > rsi_bank and combo_bank > obv_bank:
    print("\nConclusion:\tComparing Multiple Methods was the most effective method.")

plt.show()

