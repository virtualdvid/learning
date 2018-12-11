Function sigmoid(z) As Double
    sigmoid = 1 / (1 + Exp(-z))
End Function
     
Function sigmoid_p(z) As Double
    sigmoid_p = sigmoid(z) * (1 - sigmoid(z))
End Function

Sub clear()
    ActiveWorkbook.Sheets(2).Select
    Range("A2").Select
    Range(Selection, Selection.End(xlToRight)).Select
    Range(Selection, Selection.End(xlDown)).Select
    Selection.Delete
    ActiveWorkbook.Sheets(1).Select
    Range("C3:E4").Select
    Selection.ClearContents
    Range("A1").Select
End Sub

Sub main()
    Dim data As Range
    Dim point As Variant
    Dim w1, w2, b, z As Double
    Dim i, ri As Integer
    Dim Pred, Target, cost As Double
    Dim dcost_pred, dpred_dz, dcost_dz As Double
    Dim dz_dw1, dz_dw2, dz_db As Double
    Dim dcost_dw1, dcost_dw2, dcost_b As Double
    
    If ActiveWorkbook.Sheets(2).Range("A2").Value <> Empty Then
        clear
    End If
                
    Set data = Range("I3:K10")
    w1 = Rnd()
    w2 = Rnd()
    b = Rnd()
    
    Range("C3").Value = w1
    Range("D3").Value = w2
    Range("E3").Value = b
        
    For i = 1 To Range("D8").Value
        ri = Int((data.Rows.Count - 1 + 1) * Rnd + 1)
        
        point = Array(data(ri, 1).Value, data(ri, 2).Value, data(ri, 3).Value)
        
        z = point(0) * w1 + point(1) * w2 + b
        Pred = sigmoid(z)
        
        Target = point(2)
        cost = Exp(Pred - Target)
        
        dcost_pred = 2 * (Pred - Target)
        dpred_dz = sigmoid_p(z)
        
        dz_dw1 = point(0)
        dz_dw2 = point(1)
        dz_db = 1
        
        dcost_dz = dcost_pred * dpred_dz
        
        dcost_dw1 = dcost_dz * dz_dw1
        dcost_dw2 = dcost_dz * dz_dw2
        dcost_db = dcost_dz * dz_db
        
        w1 = w1 - Range("D7").Value * dcost_dw1
        w2 = w2 - Range("D7").Value * dcost_dw2
        b = b - Range("D7").Value * dcost_db
        
        Range("C4").Value = w1
        Range("D4").Value = w2
        Range("E4").Value = b
        
        ActiveWorkbook.Sheets(2).Range("A" & i + 1).Value = cost
        ActiveWorkbook.Sheets(2).Range("B" & i + 1).Value = w1
        ActiveWorkbook.Sheets(2).Range("C" & i + 1).Value = w2
        ActiveWorkbook.Sheets(2).Range("D" & i + 1).Value = b
        
        Range("D9").Value = i
        
        'Application.Wait (Now + TimeValue("0:00:05"))
        'Dim myChart As ChartObject
        'For Each myChart In ActiveSheet.ChartObjects
        '    myChart.Chart.Refresh
        '    ActiveSheet.Calculate
        'Next myChart

    Next i

End Sub