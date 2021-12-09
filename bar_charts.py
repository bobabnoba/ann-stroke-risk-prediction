import pandas as pd
from matplotlib import pyplot as plt


X = pd.read_csv(r'dataset.csv')

# # ****** S M O K I N G  S T A T U S *************************
# former = 0
# never = 0
# smoker = 0
# unknown = 0
#
# for sm in X.smoking_status.tolist():
#   if sm == "formerly smoked":
#     former += 1
#   elif sm == "never smoked":
#     never += 1
#   elif sm == "smokes":
#     smoker += 1
#   else:
#     unknown +=1
#
# print('smokers: ', smoker, '\nformer smoker:', former, '\nnon-smoker:', never, '\nunknown status:', unknown)
#
# x = ['Smoker', 'Non-smoker', 'Former smoker', 'Unknown']
# values = [smoker, former, never, unknown]

# ****** GENDER *************************
# male = 0
# female = 0
# other = 0
#
# for sm in X.gender.tolist():
#   if sm == "Male":
#     male += 1
#   elif sm == "Female":
#     female += 1
#   elif sm == "Other":
#     other += 1
#
# x = ['Female', 'Male', 'Other']
# values = [female, male, other]

# # ****** E V E R  M A R R I E D *************************
# y = 0
# n = 0
#
# for sm in X.ever_married.tolist():
#   if sm == "Yes":
#     y += 1
#   elif sm == "No":
#     n += 1
#
# x = ['married', 'not married']
# values = [y, n]

# # ****** W O R K  T Y P E *************************
# gov = 0
# priv = 0
# self = 0
# more = 0
#
# for sm in X.work_type.tolist():
#   if sm == "Govt_job":
#       gov += 1
#   elif sm == "Self-employed":
#       self += 1
#   elif sm == "Private":
#       priv += 1
#   else:
#       more += 1
#
# x = ['Govt', 'Self-empl', 'Private', 'Children']
# values = [gov, self, priv, more]

# ****** R E S I D E N C E  T Y P E  *************************
rural = 0
urban = 0
more = 0

for sm in X.Residence_type.tolist():
  if sm == "Rural":
      rural += 1
  elif sm == "Urban":
      urban += 1
  else:
      more += 1

x = ['Rural', 'Urban']
values = [rural, urban]

x_pos = [i for i, _ in enumerate(x)]
plt.bar(x_pos, values, color='red')
plt.xlabel("Status")
plt.ylabel("Number of appearances")

plt.xticks(x_pos, x)

plt.show()