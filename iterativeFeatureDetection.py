#Iterative feature detection
from sklearn.feature_selection import RFE
select = RFE (RandomForestClassifier(n_estimators=100,random_state=42),n_features_to_select=40)
select.fit(X_train,y_train)
mask = select.get_support()
plt.matshow(mask.reshape(1,-1),cmap ='gray_r')
plt.xlabel("Sample index")
plt.yticks(())
