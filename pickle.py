def f(x,y,*models):
  for m in models:
    m.fit(x,y)
    
  y_preds = [m.predict(x) for m in models]

  scores = [classification_report(y, y_pred, output_dict=True) for y_pred in y_preds]

  precisions = [score['weighted avg']['precision'] for score in scores]
  recalls = [score['weighted avg']['recall'] for score in scores]
  f1s = [score['weighted avg']['f1-score'] for score in scores]

  prec_max = np.argmax(precisions)
  recal_max = np.argmax(recalls)
  f1_max = np.argmax(f1s)

  if prec_max == recall_max and prec_max==f1_max:
    return models[f1_max]
  else:
    pass


best_model=f(x,y,LogisticRegression(),LinearDiscriminantAnalysis(),...)

filename = 'finalized_model.sav'
pickle.dump(best_model, open(filename, 'wb'))

