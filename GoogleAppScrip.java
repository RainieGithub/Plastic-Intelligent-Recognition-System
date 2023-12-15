function doGet(e) {
  var params = e.parameter;
  var data = params.data;
  var SpreadSheet = SpreadsheetApp.openByUrl('https://docs.google.com/spreadsheets/d/1vNWj2_7TKNVXkQ5XsgAO3MPqcO1FvDvhpKktMU0V5Oc/edit#gid=0'); //此處填入Google試算表的網址
  var Sheet = SpreadSheet.getSheetByName('original_data');  //此處填入試算表的標籤名稱
  //var LastRow = Sheet.getLastRow();  //資料上舊下新
  Sheet.insertRowBefore(1);        //資料上新下舊
  //寫入資料
  data = data.split(',');
  data.forEach(function(e,i){
  //Sheet.getRange(LastRow+1, i+1).setValue(e);  //(開啟資料上舊下新模式)
  Sheet.getRange(1, i+1).setValue(e);        //(開啟資料上新下舊模式)
  });
  return ContentService.createTextOutput(1);  //接收成功以後回傳"1"(類似Tingspeak)
}
