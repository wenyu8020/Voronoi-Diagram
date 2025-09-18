# Voronoi Diagram
Voronoi Diagram 是一種計算幾何中的劃分方法，它會根據一組已知的點，將平面劃分為多個區域。每個區域內的所有點，與其對應的點之間的距離都比其他點更近。

## 輸出與輸入（資料）規格

### 輸入

分成三種形式：
1. 不含註解的檔案。
2. 含註解的檔案，如一行中開頭為「#」，表示此行為註解，為不需要的資料，需做忽略。
3. 以滑鼠點擊畫布(600*600)，產生資料。

### 輸出

輸出檔案格式：
- 輸入的座標點：P x y    // 每個點佔一行，兩整數 x, y 為座標。
- 線段：E x1 y1 x2 y2    // (x1, y1) 為起點，(x2, y2) 為終點，其中 x1 ≦ x2 或 x1 = x2, y1 ≦ y2。

座標點排列在前半段，線段排列在後半段。座標點以 lexical order順序排列（即先排序第一維座標，若相同，則再排序第二維座標）；線段亦以 lexical order順序排列。

輸出文字檔案範例：

<img src="https://github.com/wenyu8020/VoronoiDiagram/blob/main/M123040016_webpage/Voronoi%20Diagram.files/image001.png">

## 功能規格與介面規格

### 介面
<img src="https://github.com/wenyu8020/VoronoiDiagram/blob/main/M123040016_webpage/Voronoi%20Diagram.files/image002.png">

### 功能說明
- Next: 下一筆測資
- Run: 執行
- Step by step: 逐步執行
- Clear: 清空畫布
- Open: 開啟與讀取檔案
- Save: 輸出檔案

## 安裝說明
下載 [Voronoi.exe](https://drive.google.com/file/d/1sU_OxxbmK34CeqRf2QVOF5RD5NA0R3ZN/view?usp=sharing) 後並執行即可根據以下步驟操作。

## 執行步驟
### 繪製點座標
- 可用滑鼠點擊畫布，即可產生點座標。
- 點擊「Open」按鈕，選擇輸入檔。

<img src="https://github.com/wenyu8020/VoronoiDiagram/blob/main/M123040016_webpage/Voronoi%20Diagram.files/image003.png">

### 繪製Voronoi Diagram
- 點擊「Run」按鈕，直接畫出Voronoi Diagram。
- 點擊「Step」按鈕，逐步執行，最後畫出Voronoi Diagram。
- 點擊「Open」按鈕，選擇輸出檔，直接根據輸出檔資料畫出Voronoi Diagram。

<img src="https://github.com/wenyu8020/VoronoiDiagram/blob/main/M123040016_webpage/Voronoi%20Diagram.files/image004.png">
<img src="https://github.com/wenyu8020/VoronoiDiagram/blob/main/M123040016_webpage/Voronoi%20Diagram.files/image006.jpg">

### 輸出結果
- 點擊「Save」按鈕，將結果輸出至檔案並儲存。

<img src="https://github.com/wenyu8020/VoronoiDiagram/blob/main/M123040016_webpage/Voronoi%20Diagram.files/image010.png">

## 實驗結果
仍有多點case的結果產生錯誤，主要是在刪線的出現問題，如果Voronoi Diagram的線段沒有錯誤，那在merge的時候就會得出正確結果，如果還有沒有刪除乾淨的線段，就會影響到接下來merge的部分。

### 3點以下

<img src="https://github.com/wenyu8020/VoronoiDiagram/blob/main/M123040016_webpage/Voronoi%20Diagram.files/image017.png">
<img src="https://github.com/wenyu8020/VoronoiDiagram/blob/main/M123040016_webpage/Voronoi%20Diagram.files/image019.png">

### 4~6點（merge 1次）

<img src="https://github.com/wenyu8020/VoronoiDiagram/blob/main/M123040016_webpage/Voronoi%20Diagram.files/image025.png">
<img src="https://github.com/wenyu8020/VoronoiDiagram/blob/main/M123040016_webpage/Voronoi%20Diagram.files/image027.png">

### 7點以上（merge 2次以上）

<img src="https://github.com/wenyu8020/VoronoiDiagram/blob/main/M123040016_webpage/Voronoi%20Diagram.files/image039.png">
<img src="https://github.com/wenyu8020/VoronoiDiagram/blob/main/M123040016_webpage/Voronoi%20Diagram.files/image041.png">

