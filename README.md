# WASSA-2021-Shared-Task

## Empathy Results:

|Model|Input|Output|Training MSE|Training Correlation|Validation MSE|Validation Correlation|
|:-------|:--------|:--------|:-------|:--------|:--------|:--------|
|RoBERTa Base Model|Essay|empathy_score|2.2722|0.5688|2.6709|0.4754|
|RoBERTa + multi input|Essay + iri_scores + personality_scores|empathy_score|1.7810|0.6896|2.5915|0.5042|
|RoBERTa + multi input|Essay + standardized iri_scores + standardized personality_scores|empathy_score|2.042|0.6302|2.6173|0.4941|
|RoBERTa + multi input|Essay + iri_scores + personality_scores + entity embeddings (age, gender, education, race, emotion)|empathy_score|1.9700|0.6474|2.59929|0.5059|
|RoBERTa + multi input|Essay + iri_scores + personality_scores + entity embeddings (age, gender, education, race)|empathy_score|1.8393|0.6838|2.5361|0.5151|
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
|RoBERTa + multi input|Essay + iri_scores + personality_scores|empathy_score|2.195|0.6399|2.8154|0.4992|
|RoBERTa + multi input|Essay + standardized iri_scores + standardized personality_scores|empathy_score|2.0843|0.6599|2.5623|0.558|
|RoBERTa + multi input|Essay + iri_scores + personality_scores + entity embeddings (age, gender, education, race, emotion)|empathy_score|2.2302|0.6409|2.8275|0.4964|
|RoBERTa + multi input|Essay + iri_scores + personality_scores + entity embeddings (age, gender, education, race)|empathy_score|2.0869|0.6463|2.6317|0.5615|
|Multi-Task Learning|Essay|empathy_bin, emotion, gender, education, age, race, empathy_score|2.4154|0.5747|3.1720|0.4130|
|Multi-Task Learning|Essay + iri + personality|empathy_bin, emotion, gender, education, age, race, empathy_score|2.0652|0.6687|2.8644|0.5007|

