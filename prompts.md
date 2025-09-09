# Planning

I want to try an idea to train a new model using a different training objective.
I want to minimise cosine distance of predictions instead of the normal one-hot token loss.
I'm trying to understand:

how should I approach this experiment? should I start training a model fresh? what size?
should I start from an existing model?
do I compare by showing my approach yields better results than normal approaches? do I simply run the same process but with an existing loss function?
can I use huggingface for this?
how do I calculate the distance? do I load BERT?
I'm new to model training and need some help.
Let's brainstorm how to setup the project.
I want to keep it simple, a couple of python files should be enough no?
What might I be missing?

> some response

hasn't the distillgpt2 seen the wikitext-2-raw-v1 data? would this not affect the experiment? Should I find new data? Maybe we can use HuggingFaceFW/finepdfs what do you think? find more info online. We can use a subset if needed.

how can I split the text too? can I still use tokens in this space?
do we need a different way to discretise it right? any ideas?

Based on the conversation create a new training-plan.md with the detailed step plan of what we need to do!
What is the theory we're using, what we're proving, the experiment setup and then how we're going to code it and how to test/validate the experiment.
Needs to be good enough to provide to a junior developer and they will be able to code it and evaluate.