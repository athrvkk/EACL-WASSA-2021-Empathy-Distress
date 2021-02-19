# WASSA-2021-Shared-Task

## Empathy Results:

|Model|Input|Output|Training MSE|Training Correlation|Validation MSE|Validation Correlation|
|:-------|:--------|:--------|:-------|:--------|:--------|:--------|
|RoBERTa Base Model|Essay|empathy_score|2.2722|0.5688|2.6709|0.4754|
|RoBERTa + multi input|Essay + iri_scores + personality_scores|empathy_score|1.7810|0.6896|2.5915|0.5042|
|Multi-Task Learning|Essay|empathy_bin, emotion, gender, education, age, race, empathy_score|1.9714|0.6403|2.6818|0.4772|
|Multi-Task Learning|Essay|empathy_bin, emotion, gender, education, race, empathy_score|2.0208|0.6345|2.6589|0.4873|
|Multi-Task Learning|Essay + sentiment score (TextBlob)|empathy_bin, emotion, gender, education, race, empathy_score|2.0034|0.6442|2.6409|0.4868|
|Multi-Task Learning|Essay + iri + personality|empathy_bin, emotion, gender, education, age, race, empathy_score|1.4403|0.7666|2.5251|0.5281|
|Multi-Task Learning|Essay + iri + personality (direct) + gender + education + race + age_bin (integer-encoded entity embeddings)|empathy_bin, emotion empathy_score|1.9357|0.6533|2.4381|0.5414|
|Multi-Task Learning|Essay + iri + personality (direct) + gender + education + race + age_bin (onehot-encoded entity embeddings)|empathy_bin, emotion empathy_score|1.555|0.7436|2.2919|0.5798|


## Distress Results:

|Model|Input|Output|Training MSE|Training Correlation|Validation MSE|Validation Correlation|
|:-------|:--------|:-------|:--------|:--------|:--------|:--------|
|RoBERTa Base Model|Essay|empathy_score| 2.4419|0.5669|3.1516|0.4130|
|RoBERTa + multi input|Essay + iri_scores + personality_scores|empathy_score|1.7810|0.6896|2.5915|0.5042|
|Multi-Task Learning|Essay|empathy_bin, emotion, gender, education, age, race, empathy_score|2.4154|0.5747|3.1720|0.4130|
|Multi-Task Learning|Essay + iri + personality|empathy_bin, emotion, gender, education, age, race, empathy_score|2.0652|0.6687|2.8644|0.5007|

