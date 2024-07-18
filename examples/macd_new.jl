using Indicators
using JSON

include("./603350.jl")

stock_new_data = JSON.parse(stock603350)
n = length(stock_new_data)

ohlcv = zeros(Float64, n, 5)

for i in 1:n
    target = stock_new_data[i]
    ohlcv[i, :] = [target["open"], target["high"], target["low"], target["close"], target["volume"]]
end

result = Indicators.macdAgu(ohlcv[:, 4])
