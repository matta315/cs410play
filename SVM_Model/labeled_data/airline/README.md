Credit [source](https://colab.research.google.com/drive/1lOcAhIffLW8XC6QsKzt5T_ZqPP4Y9eS4#scrollTo=OMC7KZd9zLk9)

More about this [dataset](https://catalog.ldc.upenn.edu/docs/LDC93S4B/corpus.html)
- [train](http://magnitude.plasticity.ai/data/atis/atis-intent-train.txt)
- [test](http://magnitude.plasticity.ai/data/atis/atis-intent-test.txt)

This dataset maps a passenger's spoken sentence to the 1 of the 17 possible intents. Examples:

When the user says something like:
* "i want to fly from boston at 838 am and arrive in denver at 1110 in the morning"*
* "what flights are available from pittsburgh to baltimore on thursday morning"*
* "show me the flights from san diego to newark by way of houston"*

=> The intent should be: atis_flight

When the user says something like:
* "what are the times that you have planes leaving from san francisco going to pittsburgh on july seventh"*
* "what is the arrival time in san francisco for the 755 am flight leaving washington"*
* "on united airlines give me the flight times from boston to dallas"*

=> The intent should be: atis_flight_time