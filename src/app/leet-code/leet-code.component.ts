import {Component, OnInit} from '@angular/core';
import {
  PriorityQueue,
  MinPriorityQueue,
  MaxPriorityQueue,
  ICompare, IGetCompareValue,
} from '@datastructures-js/priority-queue';
import {MaxHeap} from "../max-heap";
import {convertTree, TreeNode} from "../tree-node";
import {buildLinkedList, ListNode} from "../linked-list";
import {ArrayReader} from "../array-reader";
import {data3, data5} from "../data";
import {MinHeap} from "../min-heap";
import {min} from "rxjs";
import {QuickUnion} from "../quick-union";

@Component({
  selector: 'app-leet-code',
  templateUrl: './leet-code.component.html',
  styleUrls: ['./leet-code.component.scss']
})
export class LeetCodeComponent implements OnInit {
  ngOnInit(): void {
    // console.log('videoStitching',  this.videoStitching(  [[0,2],[4,6],[8,10],[1,9],[1,5],[5,9]], 10));
    // console.log('videoStitching',  this.videoStitching(  [[0,1],[1,2]], 5));
    // console.log('videoStitching',  this.videoStitching(
    //   [[0,1],[6,8],[0,2],[5,6],[0,4],[0,3],[6,7],[1,3],[4,7],[1,4],[2,5],[2,6],[3,4],[4,5],[5,7],[6,9]], 9));

    // let node1 = convertTree([3, 2, 3, null, 3, null, 1]);
    // let node2 = convertTree([3,4,5,1,3,null,1]);
    // let reader1 = new ArrayReader([7, 7, 7, 7, 10, 7, 7, 7]);
    // let reader2 = new ArrayReader([6, 6, 12]);

    // console.log('thirdMax', this.thirdMax([1,1,4,2,1,3]));
    // console.log('thirdMax', this.thirdMax([5,1,2,3,4]));
    // console.log('findDisappearedNumbers', this.findDisappearedNumbers([2,2,3,1]));
    // let head1 = buildLinkedList([1,2,4]);
    // let head2 = buildLinkedList([5]);
    // console.log('findCircleNumBfs', this.findCircleNumBfs([[1,1,0],[1,1,0],[0,0,1]]));
    // console.log('validTree', this.validTreeDfs(5, [[0,1],[0,2],[0,3],[1,4]]));
    console.log('validTree', this.validTreeUnion(5, [[0,1],[0,2],[0,3],[1,4]]));
    // console.log('validTree', this.validTreeDfs(5, [[0,1],[1,2],[2,3],[1,3],[1,4]]));
    console.log('validTree', this.validTreeUnion(5, [[0,1],[1,2],[2,3],[1,3],[1,4]]));
    console.log('validTree', this.validTreeUnion(5, [[0, 1], [1, 2], [3, 4]]));
    console.log('validTree', this.validTreeUnion(1, []));
    // console.log('validTree', this.validTreeDfs(4, [[2,3],[1,2],[1,3]]));
  }

  validTreeUnion(n: number, edges: number[][]): boolean {
    if (edges.length != n - 1) { //exactly n - 1 edges for valid tree
      return false;
    }
    const qn = new QuickUnion(n);
    for(let i = 0; i < edges.length; i++) {
      let [x, y] = edges[i];
      qn.union(x, y);
    }
    const set = new Set();
    for(let i = 0; i < n; i++) {
      let r = qn.find(i);
      set.add(r);
    }
    return set.size == 1;
  }

  validTreeDfs(n: number, edges: number[][]): boolean {
    if (edges.length != n - 1) { //exactly n - 1 edges for valid tree
      return false;
    }

    const graph = new Map();
    for(let i = 0; i < edges.length; i++) {
      let [x, y] = edges[i]
      if(!graph.has(x)) {
        graph.set(x, []);
      }
      if(!graph.has(y)) {
        graph.set(y, []);
      }
      graph.get(x).push(y);
      graph.get(y).push(x);
    }

    //detect circle and roots
    const dfsRecur = (node: number, set: Set<number>) => {
      if(set.has(node)) {// revisit a node that is already visited
        return;
      }
      set.add(node);
      for (const neighbor of graph.get(node)) {
        dfsRecur(neighbor, set);
      }
    }

    const dfsItr = (k: number): boolean => {
      const seen = new Set();
      seen.add(k);
      let queue = [k];
      while(queue.length > 0) {
        let node = queue.pop();
        for(const neighbor of graph.get(node) || []) {
          if(!seen.has(neighbor)) {
            seen.add(neighbor);
            queue.push(neighbor);
          }
        }
      }
      return seen.size == n;
    }

    let seen = new Set<number>();
    dfsRecur(0, seen)
    return seen.size == n;
  };

  validTreeBfs(n: number, edges: number[][]): boolean {
    if (edges.length != n - 1) { //exactly n - 1 edges for valid tree
      return false;
    }
    const graph = new Map();
    for(let i = 0; i < edges.length; i++) {
      let [x, y] = edges[i]
      if(!graph.has(x)) {
        graph.set(x, []);
      }
      if(!graph.has(y)) {
        graph.set(y, []);
      }
      graph.get(x).push(y);
      graph.get(y).push(x);
    }

    //detect circle and roots
    const bfs = (k: number): boolean => {
      const seen = new Set();
      seen.add(k);
      let queue = [k];
      while(queue.length > 0) {
        let nextQueue = [];
        for(let i = 0; i < queue.length; i++) {
          let node = queue[i];
          for(const neighbor of graph.get(node)||[]){
            if(!seen.has(neighbor)) {
              nextQueue.push(neighbor);
              seen.add(neighbor);
            }
          }
        }
        queue = nextQueue;
      }

      return seen.size == n;
    };

    return bfs(0);
  };

  findCircleNumBfs(isConnected: number[][]): number {
    const graph = new Map();
    const n = isConnected.length;
    const set = new Set();

    const bfs = (node: number) => {
      let queue = [node];
      set.add(node);
      while(queue.length > 0) {
        let nextQueue = [];
        for(const nextNode of graph.get(node)) {
          if(!set.has(nextNode)) {
            nextQueue.push(nextNode);
            set.add(nextNode)
          }
        }
        queue = nextQueue;
      }
    }

    for(let i = 0; i < n; i++) {
      if(!graph.has(i)) {
        graph.set(i, []);
      }
      for(let j = 0; j < isConnected[i].length; j++) {
        if(!graph.has(j)) {
          graph.set(j, []);
        }
        if (isConnected[i][j] === 1) {
          graph.get(i).push(j);
          graph.get(j).push(i);
        }
      }
    }

    let ans = 0;
    for(let i = 0; i < n; i++) {
      if(!set.has(i)) {
        ans += 1;
        bfs(i);
      }
    }

    return ans;
  }

  findCircleNumQuickUnion(isConnected: number[][]): number {
    const n = isConnected.length;
    const union = new QuickUnion(n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if(isConnected[i][j] == 1){
          union.union(i, j);
        }
      }
    }
    let set = new Set();
    for (let j = 0; j < n; j++) {
      let root = union.find(j);
      if(!set.has(root)){
        set.add(root);
      }
    }

    return set.size;
  }

  furthestBuilding(heights: number[], bricks: number, ladders: number): number {
    let sorted: number[][] = [];
    for (let i = 0; i < heights.length - 1; i++) {
      let curr = heights[i];
      let next = heights[i + 1];
      let diff = next - curr;
      if (diff > 0) {
        sorted.push([i + 1, diff]);
      }
      sorted.sort((a, b) => a[1] - b[1]);
    }
    const reachable = (end: number, br: number, ld: number): boolean => {

      for (let i = 0; i < sorted.length; i++) {
        if(sorted[i][0] > end) {
          continue;
        }
        br -= sorted[i][1];
        if (br < 0) {
          ld--;
        }
        if(br < 0 && ld < 0) {
          return false;
        }
      }
      return true;
    }

    let right = heights.length - 1, left = 0;
    let ans = 0;
    while(left <= right) {
      let mid = Math.floor((left + right) / 2);
      if(reachable(mid, bricks, ladders)) {
        ans = mid;
        left = mid + 1;
      } else {
        right = mid - 1;
      }
    }
    return ans;
  };

  kClosest(points: number[][], k: number): number[][] {
    const calc = (a: number, b: number) => a * a + b * b;
    let max = 0, min = Number.MAX_SAFE_INTEGER;
    for (let i = 0; i < points.length; i++) {
      let curr = points[i];
      let val = calc(curr[0], curr[1]);
      curr.push(val);
      if(val > max) {
        max = val;
      }
      if(val < min) {
        min = val;
      }
    }

    const count = (distance: number): any[] => {
      let val = 0;
      let temp: any[] = [];
      for(let i = 0; i < points.length; i++) {
        if(points[i][2] <= distance) {
          val++;
          temp.push({val, point: points[i]});
        }
      }
      return temp;
    }

    let left = min, right = max;
    while(left <= right) {
      let mid = (left + right) / 2;
      let t = count(mid);
      if (t.length == k) {
        return t.map((item => item.point.slice(0, 2)));
      }
      if (t.length > k) {
        right = mid - 1;
      } else {
        left = mid + 1;
      }
    }
    return [];
  };

  minMeetingRooms(intervals: number[][]): number {
    intervals.sort((a, b) => {
      if(a[0] == b[0]) {
        return b[0] - a[0];
      }
      return a[0] - b[0];
    });
    const heap: number[] = [];
    heap.push(intervals[0][1]);
    for(let i = 1; i < intervals.length; i++) {
      let [start, end] = intervals[i];
      if(start < heap[heap.length - 1]) {
        heap.push(end);
        heap.sort((a, b) => b - a);
      } else {
        heap.pop();
        heap.push(end);
        heap.sort((a, b) => b - a);
      }
    }

    return heap.length;
  };

  kthSmallest(matrix: number[][], k: number): number {
    let n = matrix.length;
    const countRange = (mid: number, pairs: number[]): number => {
      let count = 0, row = n - 1, col = 0;
      while (row >= 0 && col < n) {
        if(matrix[row][col] > mid) {
          pairs[1] = Math.min(pairs[1], matrix[row][col]);
          row--;
        } else {
          pairs[0] = Math.max(pairs[0], matrix[row][col]);
          count += row + 1;
          col++;
        }
      }
      return count;
    };

    let start = matrix[0][0], end = matrix[n-1][n-1];
    while(start < end) {
      let mid = Math.floor((end + start) / 2);
      let pair = [matrix[0][0], matrix[n-1][n-1]];
      let count = countRange(mid, pair);
      if(count == k) {
        return pair[0]; //updated within countRange function
      }
      if(count < k) {
        start = pair[1];
      } else {
        end = pair[0];
      }
    }
    return start;
  };

  kWeakestRows(mat: number[][], k: number): number[] {
    const countOnes = (nums: number[]): number => {

      let left = 0;
      let right = nums.length;
      while(left < right) {
        let mid = Math.ceil((left + right) / 2);
        if(nums[mid] == 1) {
          left = mid;
        } else {
          right = mid - 1;
        }
      }
      if(nums[left] == 1) {
        return left + 1;
      } else {
        return 0;
      }
    };

    let result: number[][] = [];
    let m = mat.length;
    for (let i = 0; i < m; i++) {
      let row = mat[i];
      let ones = countOnes(row);
      result.push([ones, i]);
    }
    result.sort((a, b) => {
      if(a[0] == b[0]) {
        return a[1] - b[1];
      }
      return a[0] - b[0];
    });
    let answer: number[] = [];
    for(let i = 0; i < k; i++) {
      answer.push(result[i][1]);
    }
    return answer;
  };

  testHeap(nums: number[]): number[] {
    let heap = new MaxHeap(nums);
    heap.print();
    // console.log(heap.peek());
    // console.log(heap.remove());
    heap.add(6)
    console.log('after adding 6', nums);
    heap.add(2)
    console.log('after adding 2', nums);
    heap.add(7)
    console.log('after adding 7', nums);

    heap.remove()
    console.log('remove top', nums);
    heap.remove()
    console.log('remove top', nums);
    heap.remove()
    console.log('remove top', nums);
    return nums;
  }

  generateTrees(n: number): Array<TreeNode | null> {
    let memo = new Map();
    const dp = (start: number, end: number): Array<TreeNode | null>  => {
      let results: Array<TreeNode | null> = [];
      if(start > end) {
        results.push(null);
        return results;
      }

      if(memo.has(`${start}_${end}`)){
        return memo.get(`${start}_${end}`);
      }

      for(let i = start; i <= end; i++) { //i is the root value
        const leftNodes = dp(start, i - 1);//left is smaller than i
        const rightNodes = dp(i + 1, end); //right is greater than i
        //each possible combinations of the left and right pairs
        for (let j = 0; j < leftNodes.length; j++) {
          for (let k = 0; k < rightNodes.length; k++) {
            let root = new TreeNode(i, leftNodes[j], rightNodes[k]);
            results.push(root);
          }
        }
      }
      memo.set(`${start}_${end}`, results);
      return results;
    };

    return dp(1, n);
  };

  kthGrammar(n: number, k: number): number {
    const search = (nodeVal: number, kth: number, row: number): number => {
      if (row == 1) {
        return nodeVal;
      }

      let nodesAtRow = Math.pow(2, row - 1);
      if (kth > nodesAtRow / 2) { //at right side of the half
        nodeVal = nodeVal == 0 ? 1 : 0;
        return search(nodeVal, kth - nodesAtRow / 2, row - 1);
      } else {//at left side of the half
        nodeVal = nodeVal == 0 ? 0 : 1;
        return search(nodeVal, kth, row - 1);
      }
    }
    return search(0, k, n);
  };

  mergeTwoLists(list1: ListNode | null, list2: ListNode | null): ListNode | null {
    const merge = (node1: ListNode | null, node2: ListNode | null): ListNode | null => {
      if(node1 == null) {
        return node2;
      } else if(node2 == null) {
        return node1;
      } else {
        if(node1.val >= node2.val) {
          node2.next = merge(node1, node2.next);
          return node2; //return smaller one to be the head for the next comparison
        } else {
          node1.next = merge(node1.next, node2);
          return node1;
        }
      }
    };

    return merge(list1, list2);
  };

  myPow(x: number, n: number): number {
    if(n < 0) {
      return 1 / this.myPow(x, n * (-1));
    }
    if(n == 0) {
      return 1;
    }
    if(n == 1) {
      return x;
    }
    if (n % 2 == 1) {
      return x * this.myPow(x * x, (n - 1) / 2);
    }
    else {
      return this.myPow(x * x, n / 2);
    }
  };

  fib(n: number): number {
    const map = new Map();
    const fibRecur = (num: number): number => {
      if(num < 2) {
        return num;
      }
      if(map.has(num)) {
        return map.get(num);
      }

      let result = fibRecur(num - 1) + fibRecur(num - 2);
      map.set(num, result);
      return result;
    }

    return fibRecur(n);
  };

  getRow(rowIndex: number): number[] {
    if(rowIndex == 0){
      return [1];
    }
    if(rowIndex == 1){
      return [1, 1];
    }
    let prevRow = this.getRow(rowIndex - 1);
    let result = new Array(rowIndex + 1).fill(0);
    result[0] = 1;
    result[result.length - 1] = 1;
    for(let i = 1; i < result.length - 1 ; i++){
      result[i] = prevRow[i - 1] + prevRow[i];
    }

    return result;
  };

  reverseList1(head: ListNode | null): ListNode | null {
    if (!head || !head.next) {
      return head;
    }
    let prev = null;
    let curr: ListNode | null = head;
    while(curr != null) {
      let temp: ListNode | null = curr.next;
      curr.next = prev;
      prev = curr;
      curr = temp;
    }

    return prev;
  };

  swapPairs(head: ListNode | null): ListNode | null {
    if(head == null || head.next == null) {
      return head;
    }
    let firstNode = head;
    let secondNode = head.next;
    firstNode.next = this.swapPairs(secondNode.next);
    secondNode.next = firstNode;
    return secondNode;
  };
  swapPairsWhile(head: ListNode | null): ListNode | null {
    if(head == null) {
      return null;
    }
    const dummy = new ListNode();
    dummy.next = head.next;
    let prevNode = dummy;
    while (head && head.next) {
      const firstNode: ListNode = head;
      const secondNode: ListNode | null = head.next;
      //this will update dummy next node only once
      prevNode.next = secondNode;
      firstNode.next = secondNode.next;
      secondNode.next = firstNode;
      //reference to dummy node next is broken after this line
      head = firstNode.next;
      prevNode = firstNode;
    }
    return dummy.next;
  };

  radixSort(nums: number[]): number[] {
    const countingSort = (arr: number[], placeVal: number) => {
      let counts = new Array(10).fill(0);
      for(let num of arr) {
        let curr = Math.floor(num / placeVal);
        counts[curr % 10] += 1;
      }
      let startingIndex = 0;
      for(let i = 0; i < counts.length; i++) {
        let count = counts[i];
        counts[i] = startingIndex;
        startingIndex += count;
      }
      let results = new Array(arr.length);
      for(const num of arr) {
        let curr = Math.floor((num) / placeVal);
        results[counts[curr % 10]] = num;
        counts[curr % 10]++;
      }
      for(let i = 0; i < results.length; i++) {
        arr[i] = results[i];
      }
    };
    let max = Math.max(...nums);
    let placeVal = 1;
    while(Math.floor(max / placeVal) > 0) {
      countingSort(nums, placeVal);
      placeVal *= 10;
    }
    return nums;
  }

  minimumAbsDifference(arr: number[]): number[][] {
    const max = Math.max(...arr);
    const shift = Math.min(...arr);
    const counts = new Array<number>(max - shift + 1).fill(0);
    for (let i = 0; i < arr.length; i++) {
      counts[arr[i] - shift] = counts[arr[i] - shift] + 1;
    }
    let startingIndex = 0;
    for (let i = 0; i < counts.length; i++) {
      let count = counts[i];
      counts[i] = startingIndex;
      startingIndex += count;
    }

    let temp: number[] = new Array(arr.length);
    for (const num of arr) {
      temp[counts[num - shift]] = num;
      counts[num - shift] = counts[num - shift] + 1;
    }

    for (let i = 0; i < arr.length; i++) {
      arr[i] = temp[i];
    }

    let results: number[][] = [];
    let minDiff = Number.MAX_SAFE_INTEGER;
    for (let i = 0; i < arr.length - 1; i++) {
      let currDiff = arr[i + 1] - arr[i];
      if (currDiff < minDiff) {
        minDiff = currDiff;
        results = [];
        results.push([arr[i], arr[i+1]]);
      } else if(currDiff == minDiff){
        results.push([arr[i], arr[i+1]]);
      }
    }

    return results;
  };

  countingSort(nums: number[]): number[] {
    const max = Math.max(...nums);
    const shift = Math.min(...nums);
    const counts = new Array(max - shift + 1).fill(0);

    for (let i = 0; i < counts.length; i++) {
      counts[nums[i] - shift] = counts[nums[i] - shift] + 1;
    }
    let startingIndex = 0;
    for (let i = 0; i < counts.length; i++) {
      let count = counts[i];
      counts[i] = startingIndex;
      startingIndex += count;
    }

    let results: number[] = new Array(nums.length);
    for (const num of nums) {
      results[counts[num - shift]] = num;
      counts[num - shift] = counts[num - shift] + 1;
    }
    return results;
  }

  findKthLargest(nums: number[], k: number): number {
    const quickSelect = (arr: number[], kth: number): number => {
      let n = arr.length;
      let mid = Math.floor(n/2);
      let midArr = [], leftArr = [], rightArr = [];
      for(let i = 0; i < n; i++) {
        if(arr[mid] == arr[i]) {
          midArr.push(arr[i]);
        } else if(arr[mid] < arr[i]){
          rightArr.push(arr[i]);
        } else {
          leftArr.push(arr[i]);
        }
      }
      if(rightArr.length >= kth){
        return quickSelect(rightArr, kth);
      } else if((rightArr.length + midArr.length) < kth){
        return quickSelect(leftArr, kth - rightArr.length - midArr.length);
      } else {
        return midArr[0];
      }
    }

    return quickSelect(nums, k);
  };

  heapSort(nums: number[]): number[] {
    const heapify = (size: number, index: number) => {
      let maxIndex = index;
      let left = 2*index + 1;
      let right = 2*index + 2;

      if(left < size && nums[left] > nums[maxIndex]) {
        maxIndex = left;
      }
      if(right < size && nums[right] > nums[maxIndex]) {
        maxIndex = right;
      }

      if(maxIndex != index) {
        [nums[maxIndex], nums[index]] = [nums[index], nums[maxIndex]];
        heapify(size, maxIndex);
      }
    };

    for(let i = Math.floor(nums.length / 2) - 1; i >= 0; i--) {
      heapify(nums.length, i);
    }

    for(let i = nums.length - 1; i >= 0; i--) {
      [nums[0], nums[i]] = [nums[i], nums[0]];
      heapify(i, 0);
    }

    return nums;
  }

  insertionSortList(head: ListNode | null): ListNode | null {
    if(!head) {
      return head;
    }
    let dummy = new ListNode();
    let curr: ListNode | null = head;
    let prev: ListNode | null = null;
    while(curr != null) {
      prev = dummy;
      //loop through new dummy headed list
      while (prev.next != null && curr.val >= prev.next.val) {
        prev = prev.next;
      }
      let temp = prev.next;
      prev.next = curr;
      let nextInput: ListNode | null = curr.next;
      curr.next = temp;

      curr = nextInput;
    }

    return dummy.next;
  };

  sortColors(nums: number[]): number[] {
    let shift = Math.min(...nums);
    let max = Math.max(...nums);

    let counts = new Array(nums.length).fill(0);
    for (let i = 0; i < nums.length; i++) {
      counts[nums[i] - shift] = counts[nums[i] - shift] + 1;
    }
    let startingIndex = 0;
    for(let i = 0; i < max + 1; i++){
      let count = counts[i];
      counts[i] = startingIndex;
      startingIndex += count;
    }
    let results = new Array(nums.length);

    for(let num of nums){
      results[counts[num - shift]] = num;
      counts[num - shift]++;
    }

    for(let i = 0; i < results.length; i++) {
      nums[i] = results[i];
    }
    return results;
  };

  findDisappearedNumbers(nums: number[]): number[] {
    let n = nums.length;
    let result: number[] = [];
    nums.sort((a, b) => a - b);
    let curr = 1;
    for(let i = 1; i < n; i++) {
      if(nums[i - 1] != nums[i]){
        nums[curr] = nums[i];
        curr++;
      }
    }

    while(nums.length > curr){
      nums.pop();
    }

    for(let i = n - 1; i >= 0; i--) {
      let val = nums[i];
      nums[val - 1] = nums[i];
    }

    for(let i = 0; i < n; i++) {
      if(nums[i] != i + 1){
        result.push(i+1);
      }
    }

    return result;
  };

  thirdMax(nums: number[]): number {
    if(nums.length < 3) {
      return Math.max(...nums);
    }
    let set = new Set();
    let sorted: number[] = [];
    nums.sort((a, b) => b - a);
    for(let i = 0; i < nums.length; i++) {
      let temp = nums[i];
      if(!set.has(temp)) {
        set.add(temp);
        sorted.push(temp);
        if(sorted.length == 3) {
          break;
        }
      }
    }
    sorted.sort((a, b) => a - b);
    return sorted.length == 3 ? sorted[0] : sorted[sorted.length-1];
  };

  heightChecker(heights: number[]): number {
    let expected = [...heights];
    this.mergeSort(expected);
    let count = 0;
    for(let i = 0; i < heights.length; i++) {
      if(heights[i] != expected[i]) {
        count++;
      }
    }

    return count;
  };

  mergeSort(arr: number[]) {
    const merge = (left: number, mid: number, right: number, tempArr: number[]) => {
      let start1 = left;
      let start2 = mid + 1;
      let l1 = mid - left + 1;
      let l2 = right - mid;

      for(let i = 0; i < l1; i++) {
        tempArr[start1 +i] = arr[start1 +i];
      }
      for(let i = 0; i < l2; i++) {
        tempArr[start2 +i] = arr[start2 +i];
      }
      let i = 0, j = 0, k = left;
      while (i < l1 && j < l2) {
        if (tempArr[start1 + i] <= tempArr[start2 + j]) {
          arr[k] = tempArr[start1 + i];
          i += 1;
        } else {
          arr[k] = tempArr[start2 + j];
          j += 1;
        }
        k += 1;
      }

      // Copy remaining elements
      while (i < l1) {
        arr[k] = tempArr[start1 + i];
        i += 1;
        k += 1;
      }
      while (j < l2) {
        arr[k] = tempArr[start2 + j];
        j += 1;
        k += 1;
      }
    };

    const sort = (left: number, right: number, tempArr: number[]) => {
      if (left >= right) {
        return;
      }
      let mid = (left + right) / 2;
      // Sort first and second halves recursively.
      sort(left, mid, tempArr);
      sort(mid + 1, right, tempArr);
      // Merge the sorted halves.
      merge(left, mid, right, tempArr);
    }

    let tempArray = new Array<number>(arr.length);
    sort(0, arr.length - 1, tempArray);
  }

  bubbleSort(arr: number[]): number[] {
    for(let i = 0; i < arr.length; i++) {
      for(let j = 0; j < arr.length - i - 1; j++) {
        let temp = arr[j];
        if(arr[j] > arr[j + 1]) {
          arr[j] = arr[j+1];
          arr[j+1] = temp;
        }
      }
    }
    return arr;
  }

  sortArrayByParity(nums: number[]): number[] {
    let l = 0, r = nums.length - 1;
    while(l < r) {
      if(nums[l] % 2 == 0) {
        l++;
      }
      if(nums[r] % 2 == 1) {
        r--;
      }
      //swap
      if(l < r && nums[l] % 2 == 1 && nums[r] % 2 == 0) {
        let temp = nums[l];
        nums[l] = nums[r];
        nums[r] = temp;
        l++;
        r--;
      }
    }
    return nums;
  };

  moveZeroes(nums: number[]): number[] {
    let curr = 0;
    for (let i = 0; i < nums.length; i++) {
      if(nums[i] != 0){
        nums[curr] = nums[i];
        curr++;
      }
    }
    for(let i = curr; i < nums.length; i++){
      nums[i] = 0;
    }
    return nums;
  }
  replaceElements(arr: number[]): number[] {
    let max = arr[arr.length-1];
    for (let i = arr.length - 1; i > 0; i--) {
      let temp =arr[i-1];
      max = Math.max(max, arr[i]);
      arr[i - 1] = max;
      max = Math.max(max, temp);
    }
    arr[arr.length-1] = -1;
    return arr;
  };

  validMountainArray(arr: number[]): boolean {
    let n = arr.length;
    if(n < 3) {
      return false;
    }

    let increasing = 0;
    let increasingStop = false;
    let decreasing = 0;
    for(let i=0; i < arr.length - 1; i++) {
      if(arr[i] == arr[i + 1]) {
        return false;
      }
      if(!increasingStop && arr[i] < arr[i + 1]) {
        increasing++;
      } else {
        increasingStop = true;
        decreasing++;
      }
    }
    if(!increasingStop) {
      return false;
    }
    return increasing + decreasing + 1 == n;
  };

  checkIfExist(arr: number[]): boolean {
    for (let i = 0; i < arr.length; i++) {
      for (let j = 0; j < arr.length; j++) {
        if(i !== j && (arr[i] == 2 * arr[j] || 2 * arr[i] == arr[j])){
          return true;
        }
      }
    }

    return false;

  };

  removeDuplicates1(nums: number[]): number {
    let currIdx = 1;
    for(let i = 1; i < nums.length; i++) {
      if(nums[i-1] != nums[i]){
        nums[currIdx] = nums[i];
        currIdx++;
      }
    }
    return currIdx;
  };

  removeElement(nums: number[], val: number): number {
    let curr = 0;
    for(let i = 0; i < nums.length; i++) {
      if(nums[i] != val){
        nums[curr] = nums[i];
        curr++;
      }
    }
    return curr;
  };

  maxScoreIndices(nums: number[]): number[] {
    let totalZeros = 0;
    let map = new Map();
    for (let i = 0; i < nums.length; i++) {
      if(nums[i] == 0) {
        totalZeros++;
      }
    }
    let i = nums.length - 1;
    let max = totalZeros;
    map.set(totalZeros, [nums.length]);
    let countOnes = 0;
    while(i >= 0) {
      if(nums[i] == 1) {
        countOnes++;
      } else {
        totalZeros--;
      }
      let temp = totalZeros + countOnes;
      if(temp >= max) {
        if(!map.has(temp)) {
          map.set(temp, []);
        }
        map.get(temp).push(i);
        max = temp;
      }
      i--;
    }

    return map.get(max)!;
  };

  findMaxConsecutiveOnes(nums: number[]): number {
    let left = 0, right = 0;
    let max = 0;
    let zeros = 1;
    while(right < nums.length) {

      if(nums[right] == 0) {
        zeros--;
      }

      while(zeros == -1) {
        if(nums[left] == 0){
          zeros++;
        }
        left++;
      }

      max = Math.max(max, right - left + 1);
      right++;
    }
    return max;
  };

  merge(nums1: number[], m: number, nums2: number[], n: number): number[] {
    let i = 0, j = 0;
    while(j < n && i < nums1.length) {
      if(nums1[i] >= nums2[j]) {
        let k = nums1.length - 1;
        while(k > i) {
          nums1[k] = nums1[k - 1];
          k--;
        }
        nums1[i] = nums2[j];
        j++;
        i++;
      } else {
        i++;
      }
    }
    while(j < n) {
      nums1[m + j] = nums2[j];
      j++;
    }

    return nums1;
  };

  duplicateZeros(arr: number[]): number[] {
    let l = arr.length;
    for(let i = 0; i < l; i++) {
      if(arr[i] == 0){
        let j = l - 1;
        while(j > i) {
          arr[j] = arr[j - 1];
          j--;
        }
        i++;
      }
    }

    return arr;
  };

  sortedSquares(nums: number[]): number[] {
    let minIndex = -1;
    for(let i = 0; i < nums.length; i++) {
      if(nums[i] >= 0 && minIndex < 0) {
        minIndex = i;
      }
      nums[i] = nums[i] * nums[i];
    }
    if(minIndex < 0) {
      nums.reverse();
      return nums;
    }
    const result = [];
    let i = minIndex;
    let j = minIndex - 1;
    while(i < nums.length && j >= 0) {
      if(nums[i] >= nums[j]) {
        result.push(nums[j]);
        j--;
      } else {
        result.push(nums[i]);
        i++;
      }
    }
    while(i < nums.length) {
      result.push(nums[i]);
      i++;
    }
    while(j >= 0) {
      result.push(nums[j]);
      j--;
    }
    return result;
  };

  minTaps(n: number, ranges: number[]): number {
    const maxReach = new Array(n + 1).fill(0);

    // Calculate the maximum reach for each tap
    for (let i = 0; i < ranges.length; i++) {
      // Calculate the leftmost position the tap can reach
      let start = Math.max(0, i - ranges[i]);
      // Calculate the rightmost position the tap can reach
      let end = Math.min(n, i + ranges[i]);

      // Update the maximum reach for the leftmost position
      maxReach[start] = Math.max(maxReach[start], end);
    }

    // Number of taps used
    let taps = 0;
    // Current rightmost position reached
    let currEnd = 0;
    // Next rightmost position that can be reached
    let nextEnd = 0;

    // Iterate through the garden
    for (let i = 0; i <= n; i++) {
      // Current position cannot be reached
      if (i > nextEnd) {
        return -1;
      }

      // Increment taps when moving to a new tap
      if (i > currEnd) {
        taps++;
        // Move to the rightmost position that can be reached
        currEnd = nextEnd;
      }
      // Update the next rightmost position that can be reached
      nextEnd = Math.max(nextEnd, maxReach[i]);
    }
    // Return the minimum number of taps used
    return taps;
  };

  earliestFullBloom(plantTime: number[], growTime: number[]): number {
    let sorted: number[][] = [];
    for(let i = 0; i < plantTime.length; i++) {
      sorted[i] = [plantTime[i], growTime[i]];
    }
    sorted.sort((a, b) => b[1] - a[1]);
    let start = 0, end = sorted[0][0] + sorted[0][1] - 1;
    for (let i = 1; i < sorted.length; i++) {
      start = start + sorted[i-1][0];
      end = Math.max(end, start + sorted[i][0] + sorted[i][1] - 1);
    }

    return end + 1;
  };

  videoStitching1(clips: number[][], time: number): number {
    const n = clips.length
    clips.sort((a,b) => a[0] - b[0])
    const dp = Array.from({length: n}, () => Array(101))
    const dfs = (e = 0, i = 0): number => {
      if (e >= time) return 0
      if (i >= n) return Infinity

      if (dp[i][e] !== undefined) return dp[i][e]

      let skip = dfs(e, i + 1)
      let take = Infinity
      if (clips[i][0] <= e) take = Math.min(dfs(Math.max(e, clips[i][1]), i + 1) + 1, take)
      return dp[i][e] = Math.min(skip, take)
    }
    const res = dfs()
    return res === Infinity ? -1 : res
  };

  videoStitching(clips: number[][], time: number): number {
    const n = clips.length;
    const memo: number[][] = Array.from({length: n}, () => Array(101).fill(-1));
    clips.sort((a, b) => a[0] - b[0]);
    //dp(i, t) = Math.min(dp(i+1, t), dp(i+1, Math.max(t, clips[i][1])) + 1);
    const dp = (i: number, endTime: number): number => {
      if(endTime >= time) {
        return 0;
      }
      if(i >= n) {
        return Infinity;
      }

      if(memo[i][endTime] != -1) {
        return memo[i][endTime];
      }

      let skip = dp(i + 1, endTime);
      let take = Infinity;
      if(endTime >= clips[i][0]) { //overlapping
        take = dp(i + 1, Math.max(endTime, clips[i][1])) + 1;
      }
      let res = Math.min(take, skip);
      memo[i][endTime] = res;
      return res;
    };

    let ans = dp(0, 0);
    return ans == Infinity ? -1 : ans;
  };

  maximumMinutes(grid: number[][]): number {
    let m = grid.length;
    let n = grid[0].length;
    const directions = [[1, 0], [0, 1], [-1, 0], [0, -1]];
    const valid = (i: number, j: number): boolean => {
      return i >= 0 && i < m && j >= 0 && j < n;
    }

    let fireInit: number[][] = []
    const fireSetOriginal = new Set<string>();
    for(let i = 0; i < m; i++) {
      for(let j = 0; j < n; j++) {
        if(grid[i][j] == 1) {
          fireInit.push([i, j]);
          fireSetOriginal.add(`${i}_${j}`)
        }
      }
    }

    const isValid = (time: number): boolean => {
      let seen = [];
      for(let k = 0; k < m; k++) {
        seen.push(new Array(n).fill(false));
      }

      const fireSet = new Set(fireSetOriginal);

      //loop t times for fire spread
      let fireQueue: number[][] = fireInit;

      for(let i = 0; i < time; i++) {
        let nextFireQueue: number[][] = [];
        for(const [i, j] of fireQueue) {
          for(const [x, y] of directions) {
            let nextX = i + x;
            let nextY = j + y;
            if(valid(nextX, nextY) && grid[nextX][nextY] == 0 && !fireSet.has(`${nextX}_${nextY}`)) {
              fireSet.add(`${nextX}_${nextY}`);
              nextFireQueue.push([nextX, nextY]);
            }
          }
        }
        fireQueue = nextFireQueue;
      }

      let queue: number[][] = [[0, 0]];

      while(queue.length > 0) {
        let nextQueue: number[][] = [];
        for(const [i, j] of queue) {
          if(i == m - 1 && j == n - 1) {
            return true;
          }

          if(fireSet.has(`${i}_${j}`)) {
            continue;
          }

          for(const [x, y] of directions) {
            let nextX = i + x;
            let nextY = j + y;
            if(valid(nextX, nextY) && grid[nextX][nextY] == 0 && !seen[nextX][nextY] && !fireSet.has(`${nextX}_${nextY}`)) {
              seen[i][j] = true;
              nextQueue.push([nextX, nextY]);
            }
          }
        }
        queue = nextQueue;

        if(queue.length > 0) {
          let nextFireQueue: number[][] = [];
          for(const [i, j] of fireQueue) {
            for(const [x, y] of directions) {
              let nextX = i + x;
              let nextY = j + y;
              if(valid(nextX, nextY) && grid[nextX][nextY] == 0 && !fireSet.has(`${nextX}_${nextY}`)) {
                fireSet.add(`${nextX}_${nextY}`);
                nextFireQueue.push([nextX, nextY]);
              }
            }
          }
          fireQueue = nextFireQueue;
        }
      }
      return false;
    };


    let result = -1;
    let l = 0, r = m * n;
    while(l <= r) {
      let m = Math.floor((l + r)/ 2);
      if(isValid(m)) {
        result = m;
        l = m + 1
      } else {
        r = m - 1;
      }
    }

    if(result == m*n && result == r) {
      return 1000000000;
    }

    return result;
  };

  maximumMinutes1(grid: number[][]): number {
    let m = grid.length;
    let n = grid[0].length;
    const directions = [[1, 0], [0, 1], [-1, 0], [0, -1]];
    const valid = (i: number, j: number): boolean => {
      return i >= 0 && i < m && j >= 0 && j < n;
    }

    const spreadFire = (t: number, seen: boolean[][]) => {
       let queue: number[][] = [];
       for(let i = 0; i< m; i++) {
         for(let j = 0; j < n; j++) {
           if(grid[i][j] == 1) {
             seen[i][j] = true;
             queue.push([i, j]);
           }
           if(grid[i][j] == 2) {
             seen[i][j] = true;
           }
         }
       }

       for(let i = 0; i < t; i++) {
         let nextQueue: number[][] = [];
         for(const [i, j] of queue) {
           for(const [x, y] of directions) {
             let nextX = i + x;
             let nextY = j + y;
             if(valid(nextX, nextY) && grid[nextX][nextY] == 0 && !seen[nextX][nextY]) {
               seen[nextX][nextY] = true;
               nextQueue.push([nextX, nextY]);
             }
           }
         }
         queue = nextQueue;
       }
    }

    const isValid = (time: number): boolean => {
      let queue: number[][] = [[0, 0]];
      let seen = [];
      for(let k = 0; k < m; k++) {
        seen.push(new Array(n).fill(false));
      }
      spreadFire(time, seen);

      while(queue.length > 0) {
        let nextQueue: number[][] = [];
        for(const [i, j] of queue) {
          if(seen[i][j]) {
            continue;
          } else {
            seen[i][j] = true;
          }

          if(i == m - 1 && j == n - 1) {
            return true;
          }
          for(const [x, y] of directions) {
            let nextX = i + x;
            let nextY = j + y;
            if(valid(nextX, nextY) && seen[nextX][nextY] == false) {
              nextQueue.push([nextX, nextY]);
            }
          }
        }
        spreadFire(1, seen);
        queue = nextQueue;
      }
      return false;
    };


    let result = -1;
    let l = 0, r = m * n;
    while(l <= r) {
      let m = Math.floor((l + r)/ 2);
      if(isValid(m)) {
        result = m;
        l = m + 1
      } else {
        r = m - 1;
      }
    }

    if(result == m*n && result == r) {
      return 1000000000;
    }

    return result;
  };

  latestDayToCross(row: number, col: number, cells: number[][]): number {
    const directions = [[1, 0], [0, 1], [-1, 0], [0, -1]];
    const valid = (i: number, j: number): boolean => {
      return i >= 0 && i < row && j >= 0 && j < col;
    }
    const isValid = (days: number): boolean => {
      let queue: number[][] = [];
      let seen = [];
      for(let i = 0; i < row; i++) {
        seen.push(new Array(col).fill(false));
      }
      for(let i = 0; i < days; i++) {
        let [x, y] = cells[i];
        seen[x - 1][y - 1] = true; //1 based index
      }
      //start from any cell on top row
      for(let i = 0; i < col; i++) {
        queue.push([0, i]);
      }

      while(queue.length > 0) {
        let nextQueue: number[][] = [];
        for(const [i, j] of queue) {
          if(seen[i][j] == false) {
            seen[i][j] = true;
          } else {
            continue;
          }
          if(i == row - 1) {
            return true;
          }
          for(const [x, y] of directions) {
            let nextX = i + x;
            let nextY = j + y;
            if(valid(nextX, nextY) && seen[nextX][nextY] == false) {
              nextQueue.push([nextX, nextY]);
            }
          }
        }
        queue = nextQueue;
      }
      return false;
    };


    let result = -1;
    let l = 0, r = cells.length;
    while(l <= r) {
      let m = Math.floor((l + r)/ 2);
      if(isValid(m)) {
        result = m;
        l = m + 1
      } else {
        r = m - 1;
      }
    }

    return result;
  };

  swimInWater(grid: number[][]): number {
    let n = grid.length;
    const directions = [[1, 0], [0, 1], [-1, 0], [0, -1]];
    const valid = (i: number, j: number): boolean => {
      return i >= 0 && i < n && j >= 0 && j < n;
    }
    const isValid = (t: number): boolean => {
      let queue = [[0,0]];
      let seen = [];
      for(let k = 0; k < n; k++) {
        seen.push(new Array(n).fill(false));
      }
      while(queue.length > 0) {
        let nextQueue: number[][] = [];
        for(const [i, j] of queue) {
          if(grid[i][j] > t) {
            return false;
          }

          if(i == n - 1 && j == n - 1) {
            return true;
          }

          for(const [x, y] of directions) {
            let nextX = i + x;
            let nextY = j + y;
            if(valid(nextX, nextY) && grid[nextX][nextY] <= t && seen[nextX][nextY] == false) {
              seen[nextX][nextY] = true;
              nextQueue.push([nextX, nextY]);
            }
          }
        }
        queue = nextQueue;
      }
      return false;
    };


    let result = -1;
    let l = 0, r = n * n;
    while(l <= r) {
      let m = Math.floor((l + r)/ 2);
      if(isValid(m)) {
        result = m;
        r = m - 1;
      } else {
        l = m + 1
      }
    }

    return result;
  };

  minDays(bloomDay: number[], m: number, k: number): number {
    let n = bloomDay.length;

    const isValid = (days: number): boolean => {
      let count = 0;
      let bouquets = m;
      for (let i = 0; i < n; i++) {
        if (bloomDay[i] <= days) {
          count++;
          if(count == k) {
            bouquets--;
            count = 0;
          }
        } else {
          //reset
          count = 0
        }
      }
      return bouquets <= 0;
    };


    let result = -1;
    let l = 0, r = Math.max(...bloomDay);
    while(l <= r) {
      let m = Math.floor((l + r)/ 2);
      if(isValid(m)) {
        result = m;
        r = m - 1;
      } else {
        l = m + 1
      }
    }

    return result;
  };

  maxRunTime(n: number, batteries: number[]): number {
    let m = batteries.length;
    let total = batteries.reduce((a, b) => a + b, 0);
    const isValid = (time: number): boolean => {
      let required = 0;
      for(let i = 0; i < m; i++) {
        required += Math.min(batteries[i], time);
      }
      return required >= n * time;
    }

    let result = 0;
    let l = 0, r = Math.floor(total / n);
    while(l <= r) {
      let m = Math.floor((l + r)/ 2);
      if(isValid(m)) {
        result = m;
        l = m + 1
      } else {
        r = m - 1;
      }
    }

    return result;
  };

  maximumTastiness(price: number[], k: number): number {
    price.sort((a, b) => a - b);
    let min = price[0];
    let max = price[price.length - 1];
    let l = 0, r = max - min;

    const isValid = (diff: number): boolean => {
      let lastPrice = price[0];
      let count = 1;// min price at index 0
      for(let i = 1; i < price.length; i++){
        if(price[i] - lastPrice >= diff){
          count++;
          lastPrice = price[i];
        }
        if(count >= k) {
          return true;
        }
      }
      return false;
    }


    let result = 0;
    while(l <= r) {
      let m = Math.floor((l + r)/ 2);
      if(isValid(m)) {
        result = m;
        l = m + 1
      } else {
        r = m - 1;
      }
    }

    return result;
  };

  maximumCandies(candies: number[], k: number): number {
    let maxCandies = Math.max(...candies);

    const calcKids = (c: number): number => {
      let total = 0;
      for(let i = 0; i < candies.length; i++) {
        if(candies[i] >= c) {
          total += Math.floor(candies[i] / c);
        }
      }
      return total;
    }

    let l = 0, r = maxCandies;
    let result: number = 0;
    //binary search right most insertion position
    while(l <= r) {
      let m = Math.floor((l + r) / 2);
      let kids = calcKids(m);
      if(kids >= k) {
        result = m;
        l = m + 1;
      } else {
        r = m - 1;
      }
    }

    return result;

  };

  minimumTime(time: number[], totalTrips: number): number {
    let minTime = Math.min(...time);
    let l = 1, r = totalTrips * minTime;

    const calcTrips = (t: number): number => {
      let sum = time.reduce((prev, curr) => {
        return prev + Math.floor(t / curr);
      }, 0);
      return sum;
    }

    while (l <= r) {
      let m = Math.floor((l + r)/ 2);
      let mTrips = calcTrips(m);
      if(mTrips >= totalTrips) {
        console.log('m', m);
        r = m - 1;
      } else {
        l = m + 1;
      }
    }
    return l;
  };

  findMaxForm(strs: string[], m: number, n: number): number {
    let size = strs.length;

    const memo = new Map();

    const count = (str: string): number[] => {
      let size = str.length;
      let ones = str.replaceAll('0', '');
      return [size - ones.length, ones.length];
    }
    const dp = (i: number, zeros: number, ones:number): number => {
      if(i == size) {
        return 0;
      }

      if(memo.has(`${i}_${zeros}_${ones}`)) {
        return memo.get(`${i}_${zeros}_${ones}`);
      }

      let [z, o] = count(strs[i]);
      let ans1 = 0;
      if (zeros + z <= m && ones + o <= n) {
        ans1 = dp(i + 1, zeros + z, ones + o) + 1;
      }
      let ans2 = dp(i + 1, zeros, ones);

      memo.set(`${i}_${zeros}_${ones}`, Math.max(ans1, ans2));

      return memo.get(`${i}_${zeros}_${ones}`);
    };

    return dp(0, 0, 0);
  };

  maximalSquare(matrix: string[][]): number {
    const m = matrix.length;
    const n = matrix[0].length;
    const memo: number[][] = [];
    for (let i = 0; i < m; i++) {
      memo.push(new Array(n).fill(-1));
    }
    let maxLen = 0;
    const dp = (i: number, j: number): number => {
      if(i < 0 || j < 0) {
         return 0;
      }

      if(memo[i][j] != -1) {
        return memo[i][j];
      }

      let ans = 0;
      if(matrix[i][j] == '0') {
        ans = 0;
      } else {
        ans = Math.min(dp(i - 1, j), dp(i - 1, j - 1), dp(i, j - 1)) + 1;
      }

      memo[i][j] = ans;
      return ans;
    };

    for(let i=m-1;i>=0;i--){
      for(let j=n-1;j>=0;j--){
        maxLen = Math.max(maxLen, dp(i,j));
      }
    }

    return maxLen * maxLen;
  };

  maximalSquare1(matrix: string[][]): number {
    const m = matrix.length;
    const n = matrix[0].length;

    const calc = (i: number, j: number): number => {
      let size = 1;
      if(matrix[i][j] == '0'){
        return 0;
      }
      let stop = false;
      while(i + size < m && j + size < n && !stop) {
        for(let k = i; k < i + size; k++) {
          if(matrix[k][j + size] == '0'){
            stop = true;
            break;
          }
        }
        for(let k = j; k < j + size; k++) {
          if(matrix[i + size][k] == '0'){
            stop = true;
            break;
          }
        }
        if(!stop) {
          size++;
        }
      }
      return size;
    };

    let max = 0;
    for(let i = 0; i < m; i++) {
      for(let j = 0; j < n; j++) {
        max = Math.max(max, calc(i, j));
      }
    }

    return max * max;
  };

  largestDivisibleSubset(nums: number[]): number[] {
    let mem = new Array(nums.length).fill(-1);
    //sort
    nums.sort((a, b) => a - b);
    let result: number[] = [];

    const dp = (currIndex: number, curr: number[], prevNum: number) => {
      if (curr.length > result.length) {
        result = [...curr];
      }

      for (let i = currIndex; i < nums.length; i++) {
        if ( curr.length > mem[i] && nums[i] % prevNum === 0) {
          mem[i] = curr.length;
          curr.push(nums[i]);
          dp(i + 1, curr, nums[i]);
          curr.pop();
        }
      }
    }

    dp(0, [], 1);
    return result;
  };

  longestArithSeqLength(nums: number[]): number {
    let max = Math.max(...nums);
    let min = Math.min(...nums);
    let maxDiff = max - min;
    let minDiff = min - max;

    const dp = (arr: number[], diff: number): number => {
      let memo = new Map()

      let max = 1;
      for(const curr of arr) {
        let prev = curr - diff;
        if(memo.has(prev)) {
          let count = memo.get(prev) + 1;
          memo.set(curr, count);
        } else {
          memo.set(curr, 1);
        }
        max = Math.max(max,  memo.get(curr));
      }

      return max;
    }

    let result = Number.MIN_SAFE_INTEGER;
    for(let i = minDiff; i <= maxDiff; i++){
      result = Math.max(result, dp(nums, i));
    }

    return result;
  };

  numRollsToTarget(n: number, k: number, target: number): number {
    const memo: number[][] = [];
    for (let i = 0; i <= n; i++) {
      memo.push(new Array(target+1).fill(-1));
    }
    const MOD = 1000000007;
    const dp = (dices: number, sum: number): number => {
      if(dices == n) {
        return sum == target ? 1 : 0;
      }

      if(memo[dices][sum] != -1) {
        return memo[dices][sum];
      }

      let ans = 0;
      for(let j = 1; j <= Math.min(k, target - sum); j++) {
        ans += dp(dices + 1, sum + j) % MOD;
      }
      ans = ans % MOD;
      memo[dices][sum] = ans;
      return ans;
    };

    return dp(0, 0);
  };

  rob3(root: TreeNode | null): number {
    let memo = new Map();
    const dp = (node: TreeNode | null): number => {
      if (node == null) {
        return 0;
      }
      if (memo.has(node)) {
        return memo.get(node);
      }

      let nextLeft1 = null, nextLeft2 = null;
      let nextRight1 = null, nextRight2 = null;
      if (node.left != null) {
        nextLeft1 = node.left.left;
        nextLeft2 = node.left.right;
      }
      if (node.right != null) {
        nextRight1 = node.right.left;
        nextRight2 = node.right.right;
      }
      let a1 = dp(nextLeft1) + dp(nextLeft2) + dp(nextRight1) + dp(nextRight2) + node.val;
      let a2 = dp(node.left) + dp(node.right);


      let ans = Math.max(a1, a2);
      memo.set(node, ans);
      return ans;
    };

    return dp(root);
  };

  wordBreak(s: string, wordDict: string[]): boolean {
    let memo = new Map()
    const dp = (i: number): boolean => {
      if (i < 0) {
        return true;
      }

      if (memo.has(i)) {
        return memo.get(i);
      }
      for (let word of wordDict) {
        let len = word.length;
        if (i + 1 - len < 0) {// word too long
          continue;
        }
        if (s.substring(i + 1 - len, i + 1) === word && dp(i - len)) {
          memo.set(i, true);
          return true;
        }
      }

      memo.set(i, false);
      return false;
    };

    return dp(s.length - 1);
  };

  change(amount: number, coins: number[]): number {
    let memo: number[][] = [];
    for (let i = 0; i <= amount; i++) {
      memo.push(new Array(coins.length).fill(-1));
    }
    const dp = (i: number, remaining: number): number => {
      if (i >= coins.length || remaining < 0) {
        return 0;
      }
      if (remaining == 0) {
        return 1;
      }

      if (memo[remaining][i] != -1) {
        return memo[remaining][i];
      }

      let ans = dp(i, remaining - coins[i]) + dp(i + 1, remaining);
      memo[remaining][i] = ans;
      return ans;
    }

    return dp(0, amount);
  };

  maxUncrossedLines(nums1: number[], nums2: number[]): number {
    const len1 = nums1.length;
    const len2 = nums2.length;
    let memo: number[][] = [];
    for (let i = 0; i < len1; i++) {
      memo.push(new Array(len2).fill(-1));
    }
    const dp = (i: number, j: number): number => {
      if (i < 0 || j < 0) {
        return 0;
      }

      if (memo[i][j] != -1) {
        return memo[i][j];
      }

      let ans = 0;
      if (nums1[i] == nums2[j]) {
        ans = dp(i - 1, j - 1) + 1;
      } else {
        ans = Math.max(dp(i - 1, j), dp(i, j - 1));
      }

      memo[i][j] = ans;
      return ans;
    };

    return dp(len1 - 1, len2 - 1);
  };

  longestSubsequence(arr: number[], difference: number): number {
    let memo: number[][] = [];
    for (let i = 0; i <= arr.length; i++) {
      memo.push(new Array(arr.length + 1).fill(-1));
    }

    const dp = (currIdx: number, prevIdx: number): number => {
      if (currIdx >= arr.length) {
        return 0;
      }

      if (memo[currIdx][prevIdx + 1] != -1) {
        return memo[currIdx][prevIdx + 1];
      }

      let opt1 = 0;
      if (prevIdx == -1 || arr[currIdx] - arr[prevIdx] == difference) {
        opt1 = dp(currIdx + 1, currIdx) + 1;
      }
      let opt2 = dp(currIdx + 1, prevIdx);

      memo[currIdx][prevIdx + 1] = Math.max(opt1, opt2);
      return memo[currIdx][prevIdx + 1];
    };


    return dp(0, -1);
  }

  longestSubsequenceDp(arr: number[], difference: number): number {
    let max = 1;
    const map = new Map();

    for (const num of arr) {
      const prevNum = num - difference;
      if (map.has(prevNum)) {
        const length = map.get(prevNum)! + 1;
        map.set(num, length);
        max = Math.max(length, max);
      } else {
        map.set(num, 1);
      }
    }

    return max;
  };

  rob2(nums: number[]): number {
    if (nums.length == 1) {
      return nums[0];
    }
    const dp = (houses: number[], i: number, memo: number[]): number => {
      if (i >= houses.length) {
        return 0;
      }

      if (memo[i] >= 0) {
        return memo[i];
      }

      let opt1 = dp(houses, i + 2, memo) + houses[i];
      let opt2 = dp(houses, i + 1, memo);
      memo[i] = Math.max(opt1, opt2);
      return memo[i];
    };

    let source1 = [...nums];
    source1.pop();
    let source2 = [...nums];
    source2.shift();
    let temp1 = dp(source1, 0, new Array(source1.length).fill(-1));
    let temp2 = dp(source2, 0, new Array(source2.length).fill(-1));
    return Math.max(temp1, temp2);
  };

  findTargetSumWays(nums: number[], target: number): number {
    const n = nums.length;
    const cache = new Map();
    const dp = (i: number, remaining: number): number => {
      if (i < 0 && remaining < 0) {
        return 0;
      }

      if (i == -1) {
        if (remaining == 0) {
          return 1;
        } else {
          return 0;
        }
      }

      if (cache.has(`${i}_${remaining}`)) {
        return cache.get(`${i}_${remaining}`);
      }

      const result = dp(i - 1, remaining - nums[i]) + dp(i - 1, remaining + nums[i]);
      cache.set(`${i}_${remaining}`, result);
      return result;
    };

    return dp(n - 1, target);
  };

  generate(numRows: number): number[][] {

    const generateNext = (curr: number[]): number[] => {
      let next: number[] = new Array(curr.length + 1).fill(1);
      for (let i = 1; i < next.length - 1; i++) {
        next[i] = curr[i - 1] + curr[i];
      }
      return next;
    };

    const dp = (rows: number): number[][] => {
      if (rows == 1) {
        return [[1]];
      }

      if (rows == 2) {
        return [[1], [1, 1]];
      }

      let prev = dp(rows - 1);
      let lastRow = prev[prev.length - 1];
      let nextRow = generateNext(lastRow);
      prev.push(nextRow);
      return prev;
    };

    return dp(numRows);
  };


  minFallingPathSumDP(matrix: number[][]): number {
    const n = matrix.length;
    let memo: number[][] = [];
    for (let i = 0; i < n; i++) {
      memo.push(new Array(n).fill(-1));
    }

    const dp = (i: number, j: number): number => {
      if (i < 0 || j < 0 || j >= n) {
        return Number.MAX_SAFE_INTEGER;
      }

      if (i == 0) {
        return matrix[i][j];
      }

      if (memo[i][j] && memo[i][j] != -1) {
        return memo[i][j];
      }

      let ans = dp(i - 1, j + 1) + matrix[i][j];
      ans = Math.min(ans, dp(i - 1, j) + matrix[i][j]);
      ans = Math.min(ans, dp(i - 1, j - 1) + matrix[i][j]);
      memo[i][j] = ans;
      return ans;
    }

    let ans = Number.MAX_SAFE_INTEGER;
    for (let i = 0; i < n; i++) {
      ans = Math.min(ans, dp(n - 1, i))
    }

    return ans;
  };

  uniquePathsWithObstacles2(obstacleGrid: number[][]): number {
    if (obstacleGrid[0][0] == 1) {
      return 0;
    }
    const m = obstacleGrid.length;
    const n = obstacleGrid[0].length;
    let dp: number[][] = [];
    for (let i = 0; i <= m; i++) {
      dp.push(new Array(n + 1).fill(0));
    }

    dp[0][0] = 1;
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        if (obstacleGrid[i][j] == 1) {
          dp[i][j] = 0;
        } else {
          if (i > 0) {
            dp[i][j] += dp[i - 1][j];
          }
          if (j > 0) {
            dp[i][j] += dp[i][j - 1];
          }

        }
      }
    }

    return dp[m - 1][n - 1];
  };


  uniquePathsTabular(m: number, n: number): number {
    let dp: number[][] = [];
    for (let i = 0; i <= m; i++) {
      dp.push(new Array(n + 1).fill(0));
    }
    dp[m - 1][n - 1] = 1;
    for (let i = m - 1; i >= 0; i--) {
      for (let j = n - 1; j >= 0; j--) {
        if (i < m - 1) {
          dp[i][j] += dp[i + 1][j];
        }
        if (j < n - 1) {
          dp[i][j] += dp[i][j + 1];
        }
      }
    }

    return dp[0][0];
  };

  uniquePaths1(m: number, n: number): number {
    let memo: number[][] = [];
    for (let i = 0; i <= m; i++) {
      memo.push(new Array(n).fill(-1));
    }

    const dp = (r: number, c: number): number => {
      if (r == m - 1 && c == n - 1) {
        return 1;
      }

      if (memo[r][c] && memo[r][c] != -1) {
        return memo[r][c];
      }

      let ans = 0;
      if (r < m) {
        ans += dp(r + 1, c);
      }
      if (c < n) {
        ans += dp(r, c + 1);
      }
      memo[r][c] = ans;
      return ans;
    };

    return dp(0, 0);
  };

  maxProfitDP(prices: number[]): number {
    let dp: number[][] = [];
    for (let i = 0; i <= prices.length + 1; i++) {
      dp.push([0, 0]);
    }


    for (let i = prices.length - 1; i >= 0; i--) {
      for (let j = 0; j < 2; j++) {
        if (j == 1) {
          dp[i][j] = Math.max(dp[i + 1][j], dp[i + 2][0] + prices[i]);
        } else {
          dp[i][j] = Math.max(dp[i + 1][j], dp[i + 1][1] - prices[i]);
        }
      }
    }

    return dp[0][0];
  };

  maxProfitT(prices: number[]): number {
    let cache: number[][] = [];
    for (let i = 0; i < prices.length; i++) {
      cache.push([-1, -1]);
    }

    const dp = (i: number, holding: boolean): number => {
      if (i >= prices.length) {
        return 0;
      }

      let holdIdx = holding ? 1 : 0;
      if (cache[i][holdIdx] != -1) {
        return cache[i][holdIdx];
      }

      let ans = dp(i + 1, holding);
      if (holding) {
        ans = Math.max(ans, dp(i + 2, false) + prices[i]);
      } else {
        ans = Math.max(ans, dp(i + 1, true) - prices[i]);
      }
      cache[i][holdIdx] = ans;
      return ans;
    };

    return dp(0, false);
  };

  maxValueOfCoins(piles: number[][], k: number): number {
    let n = piles.length;
    let cache = new Map();
    const dp = (i: number, remaining: number): number => {
      if (remaining <= 0 || i == piles.length) {
        return 0;
      }
      let memo = [];
      for (let i = 0; i < piles.length; i++) {
        memo.push(new Array(k + 1).fill(-1));
      }

      let pile = piles[i];
      let size = Math.min(pile.length, remaining);
      let ans = dp(i + 1, remaining); //skip the current pile
      let curr = 0;
      for (let j = 0; j < size; j++) {
        curr += pile[j];
        ans = Math.max(ans, curr + dp(i + 1, remaining - j - 1));
      }

      memo[i][remaining] = ans;
      return ans;
    }
    let memo = [];
    for (let i = 0; i < piles.length; i++) {
      memo.push(new Array(k + 1).fill(-1));
    }

    return dp(0, k);
  };

  maxProfitBottomUp(k: number, prices: number[]): number {
    const matrix: number[][][] = [];
    const days = prices.length;
    for (let i = 0; i <= days; i++) {
      matrix.push([]);
      for (let j = 0; j < 2; j++) {
        matrix[i].push(new Array(k + 1).fill(0));
      }
    }


    for (let i = days - 1; i >= 0; i--) {
      for (let sell = 1; sell <= k; sell++) {
        for (let holding = 0; holding < 2; holding++) {
          if (holding === 1) {
            matrix[i][holding][sell] = Math.max(matrix[i + 1][0][sell - 1] + prices[i], matrix[i + 1][holding][sell]);
          } else {
            matrix[i][holding][sell] = Math.max(matrix[i + 1][1][sell] - prices[i], matrix[i + 1][holding][sell]);
          }
        }
      }
    }
    return matrix[0][0][k];
  };

  maxProfit(k: number, prices: number[]): number {
    const matrix: number[][][] = [];
    for (let i = 0; i <= prices.length; i++) {
      matrix.push([]);
      for (let j = 0; j < 2; j++) {
        matrix[i].push(new Array(k + 1).fill(-1));
      }
    }
    const dp = (i: number, holding: number, sell: number): number => {
      if (sell == 0 || i >= prices.length) {
        return 0;
      }
      if (matrix[i][holding][sell] >= 0) {
        return matrix[i][holding][sell];
      }

      let ans = 0;
      if (holding == 0) {
        ans = Math.max(dp(i + 1, 1, sell) - prices[i], dp(i + 1, holding, sell));
      } else {
        ans = Math.max(prices[i] + dp(i + 1, 0, sell - 1), dp(i + 1, holding, sell));
      }
      matrix[i][holding][sell] = ans;
      return ans;
    };


    return dp(0, 0, k);
  };

  longestCommonSubsequence(text1: string, text2: string): number {
    let m = text1.length, n = text2.length;
    const grid = [];
    for (let i = 0; i <= m; i++) {
      grid.push(new Array(n + 1).fill(0));
    }
    for (let i = m - 1; i >= 0; i--) {
      for (let j = n - 1; j >= 0; j--) {
        if (text1[i] == text2[j]) {
          grid[i][j] = grid[i + 1][j + 1] + 1;
        } else {
          grid[i][j] = Math.max(grid[i][j + 1], grid[i + 1][j]);
        }
      }
    }

    return grid[0][0];
  };

  longestCommonSubsequenceDP(text1: string, text2: string): number {
    let m = text1.length, n = text2.length;
    const grid: number[][] = [];
    for (let i = 0; i <= m; i++) {
      grid.push(new Array(n + 1).fill(-1));
    }
    const dp = (i: number, j: number): number => {
      if (i == text1.length || j == text2.length) {
        return 0;
      }

      if (grid[i][j] >= 0) {
        return grid[i][j];
      }
      let temp = 0;
      if (text1[i] == text2[j]) {
        temp = dp(i + 1, j + 1) + 1;
      } else {
        temp = Math.max(dp(i, j + 1), dp(i + 1, j));
      }
      grid[i][j] = temp;
      return temp;
    }

    return dp(0, 0);
  };

  coinChange(coins: number[], amount: number): number {
    let max = amount + 1;
    const arr = new Array(amount + 1).fill(max);
    arr[0] = 0;
    for (let i = 1; i <= amount; i++) {
      for (let j = 0; j < coins.length; j++) {
        if (coins[j] <= i) {
          arr[i] = Math.min(arr[i], arr[i - coins[j]] + 1);
        }
      }
    }

    return arr[amount] > amount ? -1 : arr[amount];
  };

  coinChangeDP(coins: number[], amount: number): number {
    const cache = new Map();

    const dp = (remaining: number): number => {
      if (remaining < 0) {
        return -1;
      }
      if (remaining == 0) {
        return 0;
      }

      if (cache.has(remaining)) {
        return cache.get(remaining);
      }

      let min = Number.MAX_SAFE_INTEGER;
      for (let i = 0; i < coins.length; i++) {
        let ans = dp(remaining - coins[i]);
        if (ans >= 0 && ans < Number.MAX_SAFE_INTEGER) {
          min = Math.min(min, ans + 1);
        }
      }
      cache.set(remaining, min);
      return min;
    };

    let result = dp(amount);
    return result == Number.MAX_SAFE_INTEGER ? -1 : result;
  };

  minCostClimbingStairs2(costs: number[]): number {
    let n = costs.length;
    const dp = new Array(n + 1).fill(0);

    dp[0] = 0;
    dp[1] = 0;
    for (let i = 2; i <= n; i++) {
      dp[i] = Math.min(dp[i - 1] + costs[i - 1], dp[i - 2] + costs[i - 2]);
    }

    return dp[n];
  };

  climbStairs(n: number): number {
    const dp = (num: number): number => {
      if (num > n) {
        return 0;
      }
      if (num == n) {
        return 1;
      }

      return dp(num + 1) + dp(num + 2);
    }

    return dp(0);
  };

  climbStairsTopDown(n: number): number {
    if (n == 1) {
      return 1;
    }
    const dp = new Array(n + 1).fill(0);

    dp[1] = 1;
    dp[2] = 2;
    for (let i = 3; i <= n; i++) {
      dp[i] = dp[i - 1] + dp[i - 2];
    }

    return dp[n];
  };

  mostPoints(questions: number[][]): number {
    const n = questions.length;
    const dp = new Array(n + 1).fill(0);

    for (let i = n - 1; i >= 0; i--) {
      let [p, b] = questions[i];
      let ans1 = p + ((i + 1 + b) >= n ? 0 : dp[i + 1 + b]);
      let ans2 = (i + 1) >= n ? 0 : dp[i + 1];
      dp[i] = Math.max(ans1, ans2);
    }

    return dp[0];
  };

  mostPointsRecursion(questions: number[][]): number {
    const n = questions.length;
    const cache = new Array(n).fill(-1);
    const dp = (i: number): number => {
      if (i >= n) {
        return 0;
      }
      if (cache[i] >= 0) {
        return cache[i];
      }
      let [p, b] = questions[i];
      let a1 = p + dp(i + b + 1);
      let a2 = dp(i + 1);
      cache[i] = Math.max(a1, a2);
      return cache[i];
    };

    return dp(0);
  };

  lengthOfLIS_DP(nums: number[]): number {
    const cache = new Array<number>(nums.length).fill(-1);

    const dp = (j: number): number => {
      if (cache[j] > 0) {
        return cache[j];
      }

      let ans = 1;//default base case is one
      for (let i = 0; i < j; i++) {
        if (nums[j] > nums[i]) {
          ans = Math.max(ans, dp(i) + 1);
        }
      }
      cache[j] = ans;
      return ans;
    }

    let result = 0;
    for (let i = 0; i < nums.length; i++) {
      result = Math.max(result, dp(i));
    }

    return result;
  };

  lengthOfLisBS(nums: number[]): number {
    let arr = [nums[0]];

    const findIndex = (array: number[], t: number): number => {
      let l = 0, r = array.length - 1;
      while (l <= r) {
        let m = Math.floor((l + r) / 2);
        if (array[m] >= t) {
          r = m - 1;
        } else {
          l = m + 1;
        }
      }
      return l;
    };


    for (let i = 1; i < nums.length; i++) {
      if (nums[i] > arr[arr.length - 1]) {
        arr.push(nums[i]);
      } else {
        let index = findIndex(arr, nums[i]);
        arr[index] = nums[i];
      }
    }


    return arr.length;
  };

  lengthOfLIS(nums: number[]): number {
    let arr = [nums[0]];

    const findIndex = (array: number[], t: number): number => {
      let i = 0;
      for (i = 0; i < array.length; i++) {
        if (array[i] >= t) {
          break;
        }
      }
      return i;
    };


    for (let i = 1; i < nums.length; i++) {
      if (nums[i] > arr[arr.length - 1]) {
        arr.push(nums[i]);
      } else {
        let index = findIndex(arr, nums[i]);
        arr[index] = nums[i];
      }
    }


    return arr.length;
  };

  minCostClimbingStairs(costs: number[]): number {
    const n = costs.length;
    const cache: number[] = new Array(n).fill(-1);

    const dp = (i: number): number => {
      if (i == 0 || i == 1) {
        return 0;
      }

      if (cache[i] >= 0) {
        return cache[i];
      }

      const val = Math.min(dp(i - 1) + costs[i - 1], dp(i - 2) + costs[i - 2]);
      cache[i] = val;
      return val;
    }

    return dp(n);
  };

  minCostClimbingStairs1(costs: number[]): number {
    const n = costs.length;
    const minCosts: number[] = new Array(n + 1).fill(0);

    minCosts[0] = 0;
    minCosts[1] = 0;
    for (let i = 2; i <= n; i++) {
      minCosts[i] = Math.min(costs[i - 1] + minCosts[i - 1], costs[i - 2] + minCosts[i - 2]);
    }
    return minCosts[n];
  };

  robTopDown(nums: number[]): number {
    const n = nums.length;
    if (n == 1) {
      return nums[0];
    }

    let lastTwo: number = nums[0];
    let lastOne = Math.max(nums[0], nums[1]);
    for (let i = 2; i < n; i++) {
      let temp = lastOne;
      lastOne = Math.max(lastOne, lastTwo + nums[i]);
      lastTwo = temp;
    }

    return lastOne;
  };

  rob1(nums: number[]): number {
    const cache = new Map();
    const robFrom = (i: number): number => {
      if (i >= nums.length) {
        return 0;
      }
      if (cache.has(i)) {
        return cache.get(i);
      }

      let ans = Math.max(robFrom(i + 1), (nums[i] + robFrom(i + 2)))
      cache.set(i, ans);
      return ans;
    }

    return robFrom(0);
  };

  rob(nums: number[]): number {
    const cache = new Map();
    const n = nums.length;
    const robFrom = (i: number): number => {
      if (i < 0) {
        return 0;
      }

      if (i == 0) {
        return nums[0];
      }

      if (i == 1) {
        return Math.max(nums[0], nums[1]);
      }

      if (cache.has(i)) {
        return cache.get(i);
      }

      let ans = Math.max(robFrom(i - 1), (nums[i] + robFrom(i - 2)))
      cache.set(i, ans);
      return ans;
    }

    return robFrom(n - 1);
  };

  uniquePathsIII(grid: number[][]): number {
    let result = 0;
    const m = grid.length;
    const n = grid[0].length;
    const directions = [[0, 1], [0, -1], [-1, 0], [1, 0]];
    const isValid = (x: number, y: number): boolean => {
      return x >= 0 && y >= 0 && x < m && y < n;
    };
    const getKey = (x: number, y: number) => `${x}-${y}`;
    let start: number[] | null = null;
    let count = 0;
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        if (grid[i][j] == 1) {
          start = [i, j];
        } else if (grid[i][j] == 0) {
          count++;
        }
      }
    }

    const dfs = (queue: number[], emptyCount: number, seen: Set<string>) => {
      let [r, c] = queue;
      if (grid[r][c] == 2 && emptyCount == 0) {
        result++;
        return;
      }

      for (const [x1, y1] of directions) {
        let x = r + x1;
        let y = c + y1;
        if (isValid(x, y) && grid[x][y] == 2 && emptyCount == 0) {
          result++;
          return;
        }

        if (isValid(x, y) && !seen.has(getKey(x, y)) && grid[x][y] == 0) {
          let key = getKey(x, y);
          seen.add(key);
          dfs([x, y], emptyCount - 1, seen);
          seen.delete(key);
        }
      }
    };
    dfs(start!, count, new Set());
    return result;
  };

  partition(s: string): string[][] {
    const res: string[][] = [];
    dfs(s, []);
    return res;

    function dfs(str: string, path: string[]) {
      if (!str.length) {
        res.push([...path]);
        return;
      }
      for (let i = 0; i < str.length; i++) {
        const cur = str.substring(0, i + 1);
        if (isPalindrome(cur)) {
          dfs(str.substring(i + 1), [...path, cur]);
        }
      }
    }

    function isPalindrome(str: string) {
      let lo = 0,
        hi = str.length - 1;
      while (lo < hi) {
        if (str[lo++] != str[hi--]) return false;
      }
      return true;
    }
  }

  partition1(s: string): string[][] {
    const validate = (str: string): boolean => {
      let r = str.length - 1;
      let l = 0;
      while (l <= r) {
        if (str[l] !== str[r]) {
          return false;
        }
        l++;
        r--;
      }
      return true;
    };

    let result: string[][] = [];
    const backtrack = (curr: string[], str: string) => {
      if (!str.length) {
        result.push([...curr]);
        return;
      }

      for (let i = 0; i < str.length; i++) {
        let temp = str.substring(0, i + 1);// end index is exclusive
        if (validate(temp)) {
          curr.push(temp);
          backtrack(curr, str.substring(i + 1));
          curr.pop();
        }
      }
    };

    backtrack([], s);

    return result;
  };

  partition2(s: string): string[][] {
    const n = s.length;
    const getKey = (arr: string[]) => arr.join('-');
    const seen = new Set();

    const validate = (str: string): boolean => {
      let r = str.length - 1;
      let l = 0;
      while (l <= r) {
        if (str[l] !== str[r]) {
          return false;
        }
        l++;
        r--;
      }
      return true;
    };

    let result: string[][] = [];

    const backtrack = (curr: string[], start: number) => {

      const key = getKey(curr);

      if (curr.join('').length == n && !seen.has(key)) {
        seen.add(key);
        result.push([...curr]);
        return;
      }

      if (start > n) {
        return;
      }

      for (let size = 1; size <= n; size++) {
        let str = s.slice(start, start + size);// end index is exclusive
        if (validate(str)) {
          curr.push(str);
          backtrack(curr, start + size);
          curr.pop();
        }
      }
    };

    backtrack([], 0);

    return result;
  };

  restoreIpAddresses(s: string): string[] {
    let result: string[] = [];

    const backtrack = (curr: string[], start: number) => {

      if (curr.length == 4) {
        let totalLength = curr.reduce((prev, curr) => {
          return prev + curr.length;
        }, 0);
        if (totalLength == s.length) {
          let newIp = curr.join('.');
          if (!result.includes(newIp)) {
            result.push(newIp);
          }
        }
        return;
      }

      for (let size = 1; size <= 3; size++) {
        let l = s.slice(start, start + size);
        if (l.length > 1 && l.startsWith('0')) {
          return;
        }
        let val = parseInt(l);
        if (val <= 255) {
          curr.push(l);
          backtrack(curr, start + size);
          curr.pop();
        }
      }
    };

    backtrack([], 0);

    return result;
  };

  findSubsequences(nums: number[]): number[][] {
    const results: number[][] = [];
    const seen = new Set();
    const getKey = (arr: number[]) => arr.join('-');

    const backtrack = (curr: number[], j: number) => {
      // if we have checked all elements
      const key = getKey(curr);
      if (j == nums.length) {
        if (curr.length >= 2 && !seen.has(key)) {
          results.push([...curr]);
          seen.add(key);
        }
        return;
      }
      // if the sequence remains increasing after appending nums[index]
      if (curr.length == 0 || curr[curr.length - 1] <= nums[j]) {
        curr.push(nums[j]);
        backtrack(curr, j + 1);
        curr.pop();
      }
      // call recursively without appending an element
      backtrack(curr, j + 1);


    };
    backtrack([], 0);
    return results;
  };

  getHappyString(n: number, k: number): string {
    const arr = ['a', 'b', 'c'];
    let result: string[] = [];

    const backtrack = (curr: string[]) => {
      if (curr.length == n) {
        result.push(curr.join(''));
        return;
      }

      for (let i = 0; i < arr.length; i++) {
        if (curr.length == 0 || curr[curr.length - 1] != arr[i]) {
          curr.push(arr[i]);
          backtrack(curr);
          curr.pop();
        }
      }
    };

    backtrack([]);
    result.sort((a, b) => a.localeCompare(b));
    return result.length >= k ? result[k - 1] : '';
  };

  distributeCookies1(cookies: number[], k: number): number {
    let sums: number[] = [];
    let n = cookies.length;
    cookies.sort((a, b) => a - b);
    let initArray: number[][] = [];
    for (let i = 0; i < k; i++) {
      initArray.push([]);
    }

    const backtrack = (dist: number[][], cookieIndex: number, emptyCount: number) => {
      if (emptyCount > n - cookieIndex || cookieIndex > n) {
        return;
      }

      if (cookieIndex == n && emptyCount == 0) {
        let count = 0;
        let sumMax = 0;
        for (const arr of dist) {
          count += arr.length;
          let sum = arr.reduce((prev, curr) => prev + curr, 0);
          sumMax = Math.max(sumMax, sum);
        }
        if (count == n) {
          sums.push(sumMax);
        }
        return;
      }

      for (let j = 0; j < k; j++) {
        let d = dist[j];
        if (d.length == 0) {
          emptyCount--;
        }
        for (let i = cookieIndex; i < n; i++) {
          if (cookieIndex >= i || cookies[i - 1] != cookies[i]) {
            d.push(cookies[i]);
            backtrack(dist, i + 1, emptyCount);
            d.pop();
          }
        }
        if (d.length == 0) {
          emptyCount++;
        }
      }
    }
    backtrack(initArray, 0, k);
    return Math.min(...sums);
  };

  distributeCookies(cookies: number[], k: number): number {
    let sums: number[] = [];
    let n = cookies.length;
    cookies.sort((a, b) => a - b);
    let initArray: number[][] = [];
    for (let i = 0; i < k; i++) {
      initArray.push([]);
    }

    const backtrack = (dist: number[][], cookieIndex: number, emptyCount: number) => {
      if (emptyCount > n - cookieIndex || cookieIndex > n) {
        return;
      }

      if (cookieIndex == n && emptyCount == 0) {
        let count = 0;
        let sumMax = 0;
        for (const arr of dist) {
          count += arr.length;
          let sum = arr.reduce((prev, curr) => prev + curr, 0);
          sumMax = Math.max(sumMax, sum);
        }
        if (count == n) {
          sums.push(sumMax);
        }
        return;
      }

      for (let j = 0; j < k; j++) {
        let d = dist[j];
        if (d.length == 0) {
          emptyCount--;
        }
        d.push(cookies[cookieIndex]);
        backtrack(dist, cookieIndex + 1, emptyCount);
        d.pop();
        if (d.length == 0) {
          emptyCount++;
        }
      }
    }
    backtrack(initArray, 0, k);
    return Math.min(...sums);
  };

  combinationSum2(candidates: number[], target: number): number[][] {
    const results: number[][] = [];
    candidates.sort((a, b) => a - b);
    const backtrack = (curr: number[], sum: number, idx: number) => {
      console.log('backtrack', `curr=${curr}, sum=${sum}, idx=${idx}`);
      if (sum > target) {
        return;
      }
      if (sum == target) {
        results.push([...curr]);
        return;
      }

      for (let i = idx; i < candidates.length && target - sum >= candidates[i]; i++) {
        let num = candidates[i];
        //remove duplicates for last two
        if (i > idx && candidates[i - 1] == candidates[i]) {
          console.log(`Remove duplicates i=${i}, idx=${idx}, candidates[i - 1]=${candidates[i - 1]}, candidates[i]=${candidates[i]}`)
        }

        if (i == idx || (i > 0 && candidates[i - 1] != candidates[i])) {
          curr.push(num);
          console.log('pushed ', num, 'into array', curr);
          backtrack(curr, sum + num, i + 1);
          let temp = curr.pop();
          console.log('popped ', temp, 'out of array', curr);
        }
      }
    };

    backtrack([], 0, 0);

    return results;
  };

  permuteUnique(nums: number[]): number[][] {
    const results: number[][] = [];
    const map = new Map();
    for (let i = 0; i < nums.length; i++) {
      let num = nums[i];
      if (!map.has(num)) {
        map.set(num, 0);
      }
      map.set(num, map.get(num)! + 1);
    }

    const backtrack = (curr: number[]) => {
      if (curr.length == nums.length) {
        results.push([...curr]);
        return;
      }

      for (const num of map.keys()) {
        if (map.get(num) > 0) {
          map.set(num, map.get(num)! - 1);
          curr.push(num);
          backtrack(curr);
          curr.pop();
          map.set(num, map.get(num)! + 1);
        }
      }
    };

    backtrack([]);
    return results;
  };

  permuteUnique1(nums: number[]): number[][] {
    const results: number[][] = [];
    const set = new Set();
    const getKey = (arr: number[]): string => {
      return arr.join('_');
    }
    const backtrack = (curr: number[], seen: Set<number>, idx: number) => {
      let key = getKey(curr);
      if (curr.length == nums.length && !set.has(key)) {
        set.add(key);
        results.push([...curr]);
        return;
      }

      for (let i = 0; i < nums.length; i++) {
        let num = nums[i];
        if (!seen.has(i)) {
          seen.add(i);
          curr.push(num);
          backtrack(curr, seen, idx + 1);
          curr.pop();
          seen.delete(i);
        }
      }
    };

    backtrack([], new Set(), 0);
    return results;
  };

  combinationSum3(k: number, n: number): number[][] {
    const results: number[][] = [];
    const backtrack = (curr: number[], sum: number, idx: number) => {
      if (curr.length == k && sum == n) {
        results.push([...curr]);
        return;
      }
      if (sum > n) {
        return;
      }

      for (let i = idx; i <= 9; i++) {
        if (sum < n) {
          curr.push(i);
          backtrack(curr, sum + i, i + 1);
          curr.pop();
        } else {
          break;
        }
      }
    };

    backtrack([], 0, 1);
    return results;
  };

  numsSameConsecDiff(n: number, k: number): number[] {
    const results: number[] = [];
    const dfs = (curr: number[]) => {
      if (curr.length == n) {
        results.push(parseInt(curr.join('')));
        return;
      }

      for (let j = 0; j <= 9; j++) {
        let copy = [...curr];
        let diff = Math.abs(j - copy[copy.length - 1]);
        if (diff == k && copy.length < n) {
          copy.push(j);
          dfs(copy);
        }
      }

    }

    for (let i = 1; i <= 9; i++) {
      dfs([i]);
    }

    return results;
  }

  numsSameConsecDiffBfs(n: number, k: number): number[] {
    const results: number[] = [];
    const bfs = (nums: number[][]) => {
      let queue = nums;
      while (queue.length > 0) {
        let nextQueue: number[][] = [];
        for (let i = 0; i < queue.length; i++) {
          let items = queue[i];
          if (items.length >= n) {
            break;
          }
          for (let j = 0; j <= 9; j++) {
            let newItems = [...items];
            let diff = Math.abs(j - newItems[newItems.length - 1]);
            if (diff == k) {
              newItems.push(j);
              nextQueue.push(newItems);
              if (newItems.length == n) {
                results.push(parseInt(newItems.join('')));
              }
            }
          }
        }
        queue = nextQueue;
      }
    }

    let init: number[][] = [];
    for (let i = 1; i <= 9; i++) {
      init.push([i]);
    }

    bfs(init);
    return results;
  }

  numsSameConsecDiffBT(n: number, k: number): number[] {
    const results: number[] = [];
    const backtrack = (curr: number[]) => {
      if (curr.length == n) {
        results.push(parseInt(curr.join('')));
        return;
      }
      for (let i = 0; i <= 9; i++) {
        if (curr.length == 0) {
          if (i != 0) {
            curr.push(i);
            backtrack(curr);
            curr.pop();
          }
        } else {
          let diff = Math.abs(i - curr[curr.length - 1]);
          if (diff == k) {
            curr.push(i);
            backtrack(curr);
            curr.pop();
          }
        }
      }
    }

    backtrack([]);
    return results;
  };

  generateParenthesis(n: number): string[] {
    const generate = (num: number): string[] => {
      if (num === 0) {
        return [""];
      }
      let answer: string[] = [];
      for (let leftCount = 0; leftCount < num; ++leftCount) {
        for (let leftString of generate(leftCount)) {
          for (let rightString of generate(num - 1 - leftCount)) {
            answer.push("(" + leftString + ")" + rightString);
          }
        }
      }
      return answer;
    };

    return generate(n);
  }

  generateParenthesisBT(n: number): string[] {
    let results: string[] = [];
    const backtrack = (curr: string[], open: number, close: number) => {
      if (curr.length == 2 * n) {
        results.push(curr.join(''));
        return;
      }
      let options = [];
      if (open < n) {
        options.push('(');
      }
      if (close < n && close < open) {
        options.push(')');
      }

      for (let i = 0; i < options.length; i++) {
        curr.push(options[i]);
        if (options[i] == '(') {
          open++;
        } else {
          close++;
        }
        backtrack(curr, open, close);
        if (options[i] == '(') {
          open--;
        } else {
          close--;
        }
        curr.pop();
      }
    }

    backtrack(['('], 1, 0);

    return results;
  };

  exist(board: string[][], word: string): boolean {
    let m = board.length;
    let n = board[0].length;
    if (word.length == 1 && board[0][0] == word) {
      return true;
    }
    let map = new Map();
    let seen: boolean[][] = [];
    for (let i = 0; i < m; i++) {
      seen.push(new Array(n).fill(false));
      for (let j = 0; j < n; j++) {
        let l = board[i][j];
        if (!map.has(l)) {
          map.set(l, []);
        }
        map.get(l).push([i, j]);
      }
    }

    const directions = [[0, 1], [1, 0], [0, -1], [-1, 0]];
    const isValid = (r: number, c: number) => {
      return r >= 0 && c >= 0 && r < m && c < n;
    };
    const backtrack = (remaining: string[], row: number, col: number): boolean => {
      if (remaining.length == 0) {
        return true;
      }
      let w = board[row][col];
      if (remaining[0] != w || seen[row][col]) {
        return false;
      }
      seen[row][col] = true;
      for (const [x, y] of directions) {
        let nextX = row + x;
        let nextY = col + y;
        if (isValid(nextX, nextY) && backtrack(remaining.slice(1), nextX, nextY)) {
          return true;
        }
      }
      seen[row][col] = false;

      return false;
    }
    let words = word.split('');
    if (!map.has(word[0])) {
      return false;
    }
    for (const [r, c] of map.get(word[0])) {
      let res = backtrack(words, r, c);
      if (res) {
        return true;
      }
    }
    return false;
  }

  totalNQueens(n: number): number {
    if (n == 1) {
      return 1;
    }

    let result: number[][][] = [];
    const backtrack = (curr: number[][], row: number, column: Set<number>, diagonal: Set<number>, antiDiagonal: Set<number>) => {
      if (curr.length == n) {
        result.push([...curr]);
        return;
      }

      for (let j = 0; j < n; j++) {
        if (!column.has(j) && !diagonal.has(row - j) && !antiDiagonal.has(row + j)) {
          column.add(j);
          diagonal.add(row - j);
          antiDiagonal.add(row + j);
          curr.push([row, j]);
          backtrack(curr, row + 1, column, diagonal, antiDiagonal);
          column.delete(j);
          diagonal.delete(row - j);
          antiDiagonal.delete(row + j);
          curr.pop();
        }
      }
    }

    backtrack([], 0, new Set(), new Set(), new Set());
    console.log(result);
    return result.length;
  };

  combinationSum(candidates: number[], target: number): number[][] {
    const backtrack = (curr: number[], sum: number, idx: number) => {
      if (sum == target) {
        result.push([...curr]);
        return;
      }
      if (sum > target) {
        return;
      }

      for (let i = idx; i < candidates.length; i++) {
        let c = candidates[i];
        curr.push(c);
        backtrack(curr, sum + c, i);
        curr.pop();
      }
    }

    let result: number[][] = [];
    backtrack([], 0, 0);
    return result;
  };

  letterCombinations(digits: string): string[] {
    const map = new Map()
    map.set('2', ['a', 'b', 'c']);
    map.set('3', ['d', 'e', 'f']);
    map.set('4', ['g', 'h', 'i']);
    map.set('5', ['j', 'k', 'l']);
    map.set('6', ['m', 'n', 'o']);
    map.set('7', ['p', 'q', 'r', 's']);
    map.set('8', ['t', 'u', 'v']);
    map.set('9', ['w', 'x', 'y', 'z']);

    let n = digits.length;
    if (n == 0) {
      return [];
    }
    let results: string[] = [];
    const combine = (curr: string[], idx: number) => {
      if (idx > n) {
        return;
      }

      if (curr.length == n) {
        results.push(curr.join(''));
        return;
      }

      for (let i = idx; i < n; i++) {
        let d = digits[i];
        let items = map.get(d); //'abc', 'cde'
        for (let j = 0; j < items.length; j++) {
          curr.push(items[j]);
          combine(curr, i + 1);
          curr.pop();
        }
      }
    };

    combine([], 0);

    return results;
  };

  allPathsSourceTarget(graph: number[][]): number[][] {
    const result: number[][] = [];
    const n = graph.length;

    let queue: number[][] = [graph[0]];
    let curr: number[] = [0];
    while (queue.length > 0) {
      let items = queue.pop()!;
      for (let j = 0; j < items.length; j++) {
        let next = items[j];
        curr.push(next);
        if (next == n - 1) {
          result.push([...curr]);
          curr = [];
        }
        queue.push(graph[next]);
      }
    }

    return result;
  };

  combine(n: number, k: number): number[][] {
    let result: number[][] = [];
    const backtrack = (curr: number[], idx: number) => {
      if (idx > n + 1) {
        return;
      }

      if (curr.length == k) {
        result.push([...curr]);
        return;
      }

      for (let i = idx; i <= n; i++) {
        curr.push(i);
        backtrack(curr, i + 1);
        curr.pop();
      }
    }

    backtrack([], 1);
    return result;
  };

  subsets(nums: number[]): number[][] {
    let result: number[][] = [];
    const n = nums.length;
    const backtrack = (curr: number[], idx: number) => {
      if (idx > n) {
        return;
      }

      result.push([...curr]);
      for (let i = idx; i < n; i++) {
        curr.push(nums[i]);
        backtrack(curr, i + 1);
        curr.pop();
      }
    }

    backtrack([], 0);
    return result;
  };

  permute(nums: number[]): number[][] {
    let result: number[][] = [];
    const n = nums.length;
    const backtrack = (curr: number[], idx: number) => {
      if (curr.length == n) {
        result.push([...curr]);
        return;
      }

      for (let i = 0; i < n; i++) {
        let num = nums[i];
        if (!curr.includes(num)) {
          curr.push(num);
          backtrack(curr, idx + 1);
          curr.pop();
        }
      }
    }

    backtrack([], 0);
    return result;
  };

  fullBloomFlowers(flowers: number[][], people: number[]): number[] {
    let starts: number[] = [];
    let ends: number[] = [];
    for (const [s, e] of flowers) {
      starts.push(s);
      ends.push(e);
    }

    starts.sort((a: number, b: number) => a - b);
    ends.sort((a: number, b: number) => a - b);

    const search = (t: number): number => {
      let l = 0, r = starts.length;
      while (l < r) {
        let m = Math.floor((l + r) / 2);
        if (starts[m] > t) { //right most insertion point
          r = m;
        } else {
          l = m + 1;
        }
      }

      let l1 = 0, r1 = ends.length;
      while (l1 < r1) {
        let m = Math.floor((l1 + r1) / 2);
        if (ends[m] >= t) { //left most insertion point
          r1 = m;
        } else {
          l1 = m + 1;
        }
      }

      return l - l1;

    };

    let result = [];
    for (const p of people) {
      result.push(search(p));
    }

    return result;
  };


  search(nums: number[], target: number): number {
    let n = nums.length;
    let l = 0, r = nums.length - 1;
    let min = -1;
    while (l < r) {
      let m = Math.floor((l + r) / 2);
      if (nums[m] < nums[n - 1]) {
        r = m;
      } else {
        l = m + 1;
      }
    }
    min = l;
    nums = nums.slice(min).concat(nums.slice(0, min));
    l = 0;
    r = n - 1;
    while (l <= r) {
      let m = Math.floor((l + r) / 2);
      if (nums[m] == target) {
        return (m + min) % n;
      }
      if (nums[m] > target) {
        r = m - 1;
      } else {
        l = m + 1;
      }
    }
    return -1;

  };

  singleNonDuplicate(nums: number[]): number {
    if (nums.length == 1) {
      return nums[0];
    }
    let l = 0, r = nums.length - 1;
    while (l < r) {
      let m = Math.floor((l + r) / 2);
      if (nums[m] < nums[m + 1] && nums[m] > nums[m - 1]) {
        return m;
      }
      if (m % 2 == 0) {
        if (nums[m] == nums[m + 1]) {
          l = m + 2;
        } else {
          r = m - 2;
        }
      } else {
        if (nums[m] == nums[m + 1]) {
          r = m - 1;
        } else {
          l = m + 1;
        }
      }
    }
    return l;
  };

  getIndex1(reader: ArrayReader): number {
    let left = 0, length = reader.length();
    while (length > 1) {
      length = Math.floor(length / 2);
      const result = reader.compareSub(left, left + length - 1, left + length + 1, left + length + length);
      if (result == 0) {
        return left + length;
      }
      if (result < 0) {
        left = left + length + 1;
      }
    }
    return left;
  };

  getIndex(reader: ArrayReader): number {
    let left = 0, length = reader.length();
    while (length > 1) {
      length = Math.floor(length / 2);
      const result = reader.compareSub(left, left + length - 1, left + length, left + length + length - 1);
      if (result == 0) {
        return left + length + length;
      }
      if (result < 0) {
        left = left + length;
      }
    }
    return left;
  };

  countRectangles(rectangles: number[][], points: number[][]): number[] {
    const map = new Map<number, number[]>();
    for (let [l, h] of rectangles) {
      if (!map.has(h)) {
        map.set(h, []);
      }
      map.get(h)!.push(l);
    }
    let sorted: { height: number, val: number[] }[] = [];
    for (const height of map.keys()) {
      map.get(height)!.sort((l1, l2) => l1 - l2);
      sorted.push({height, val: map.get(height)!});
    }
    sorted.sort((a: { height: number, val: number[] }, b: { height: number, val: number[] }) => a.height - b.height);

    const bsY = (arr: { height: number, val: number[] }[], y: number): number => {
      let l = 0, r = arr.length;
      while (l < r) {
        let m = Math.floor((l + r) / 2);
        if (arr[m].height >= y) {
          r = m;
        } else {
          l = m + 1;
        }
      }
      return l;
    }

    const bsX = (arr: number[], x: number): number => {
      let l = 0, r = arr.length;
      while (l < r) {
        let m = Math.floor((l + r) / 2);
        if (arr[m] >= x) {
          r = m;
        } else {
          l = m + 1;
        }
      }
      return l;
    }

    let results: number[] = [];
    for (let i = 0; i < points.length; i++) {
      let [x, y] = points[i];
      let yIndex = bsY(sorted, y);
      let count = 0;
      for (let j = yIndex; j < sorted.length; j++) {
        let foundIdx = bsX(sorted[j].val, x);
        count += sorted[j].val.length - foundIdx;
      }
      results.push(count);
    }
    return results;
  };

  closestNodes(root: TreeNode | null, queries: number[]): number[][] {
    if (!root || queries.length == 0) {
      return [];
    }

    let result: number[][] = [];
    let arr: number[] = [];
    const travel = (node: TreeNode | null) => {
      if (!node) {
        return;
      }
      travel(node.left);
      arr.push(node.val);
      travel(node.right);
    }

    travel(root);
    console.log('array', arr);

    const search = (n: number): number[] => {
      let l = 0, r = arr.length - 1;
      let min = -1, max = -1;
      while (l <= r) {
        let m = Math.floor((l + r) / 2);
        if (arr[m] == n) {
          return [n, n];
        }
        if (arr[m] > n) {
          r = m - 1;
          max = arr[m];
          console.log('target, max', n, arr[m]);
        } else {
          l = m + 1;
          min = arr[m];
          console.log('target, min', n, arr[m]);
        }
      }
      return [min, max];
    }

    for (let i = 0; i < queries.length; i++) {
      console.log('search for', queries[i]);
      let item = search(queries[i]);
      console.log('search for', queries[i], 'answer', item);

      result.push(item);
    }

    return result;
  };

  maxDistance(nums1: number[], nums2: number[]): number {
    let i = 0, j = 0;
    let max = 0;
    while (i <= j && i < nums1.length && j < nums2.length) {
      if (nums2[j] >= nums1[i]) {
        max = Math.max(max, j - i);
        j++;
      } else {
        j++;
        if (i < j) {
          i++;
        }
      }
    }
    return max;
  };

  maxDistanceBs(nums1: number[], nums2: number[]): number {
    const check = (m: number): boolean => {
      for (let j = m; j < nums2.length; j++) {
        let i = j - m;
        if (i >= 0 && i < nums1.length && nums2[j] >= nums1[i]) {
          return true;
        }
      }
      return false;
    }

    let l = 0, r = nums2.length - 1;
    while (l <= r) {
      let m = Math.floor((l + r) / 2);
      if (!check(m)) {
        r = m - 1;
      } else {
        l = m + 1;
      }
    }
    return r > 0 ? r : 0;
  };

  splitArray(nums: number[], k: number): number {
    const check = (t: number): number => {
      let count = 0;
      let sum = 0;
      for (let i = 0; i < nums.length; i++) {
        if (sum + nums[i] <= t) {
          sum += nums[i];
        } else {
          count++;
          sum = nums[i];
        }
      }
      return count + 1; //mid value too small
    }
    let sum = 0;
    for (let i = 0; i < nums.length; i++) {
      sum += nums[i];
    }
    let max = 0;
    let l = Math.max(...nums), r = sum;
    while (l <= r) {
      let m = Math.floor((l + r) / 2);
      if (check(m) <= k) {
        r = m - 1;
        max = m;
      } else {
        l = m + 1;
      }
    }
    return max;
  };

  maximizeSweetness(sweetness: number[], k: number): number {
    const calc = (t: number): boolean => {
      let j = 0;
      for (let i = 0; i <= k; i++) {
        let sum = 0;
        while (sum < t) {
          if (j < sweetness.length) {
            sum += sweetness[j];
            j++;
          } else {
            return true;
          }
        }
      }
      return false;
    };

    let max = 0;
    for (let i = 0; i < sweetness.length; i++) {
      max += sweetness[i];
    }

    let l = 1, r = Math.ceil(max / (k + 1));

    while (l <= r) {
      let m = Math.floor((l + r) / 2);
      if (calc(m)) {
        r = m - 1;
      } else {
        l = m + 1;
      }
    }

    return r;
  };

  smallestDivisor(nums: number[], threshold: number): number {
    const calc = (s: number): number => {
      let t = 0;
      for (let i = 0; i < nums.length; i++) {
        t += Math.ceil(nums[i] / s);
      }
      return t;
    };

    let l = 1, r = 1000000;

    while (l <= r) {
      let m = Math.floor((l + r) / 2);
      if (calc(m) <= threshold) {
        r = m - 1;
      } else {
        l = m + 1;
      }
    }

    return l;
  };

  minSpeedOnTime(dist: number[], hour: number): number {
    const calc = (s: number): number => {
      let t = 0;
      for (let i = 0; i < dist.length; i++) {
        if (i != dist.length - 1) {
          t += Math.ceil(dist[i] / s);
        } else {
          t += dist[i] / s;
        }
      }
      return t;
    };
    let sum = 0;
    for (let i = 0; i < dist.length; i++) {
      sum += dist[i];
    }

    if (dist.length - 1 >= hour) {
      return -1;
    }

    let time = hour - dist.length + 1;
    let l = 1, r = Math.ceil(sum / time);

    while (l <= r) {
      let m = Math.floor((l + r) / 2);
      if (calc(m) <= hour) {
        r = m - 1;
      } else {
        l = m + 1;
      }
    }

    return l;
  };

  minimumEffortPathDfs(heights: number[][]): number {
    const m = heights.length;
    const n = heights[0].length;
    let max = 0, min = Infinity;
    for (let i = 0; i < m; i++) {
      max = Math.max(max, ...heights[i]);
      min = Math.min(min, ...heights[i]);
    }
    const isValid = (x: number, y: number) => {
      return x >= 0 && y >= 0 && x < m && y < n;
    };
    const directions = [[0, 1], [0, -1], [-1, 0], [1, 0]];

    const check = (mid: number) => {
      const seen = [];
      for (let i = 0; i < m; i++) {
        seen.push(new Array(n).fill(false));
      }
      seen[0][0] = true;
      return checkDfs(mid, 0, 0, seen);
    }

    const checkDfs = (k: number, x: number, y: number, seen: boolean[][]): boolean => {
      if (x == m - 1 && y == n - 1) {
        return true;
      }
      for (const [a, b] of directions) {
        let [nextX, nextY] = [x + a, y + b];
        if (isValid(nextX, nextY) && !seen[nextX][nextY]) {
          let diff = Math.abs(heights[x][y] - heights[nextX][nextY]);
          if (diff <= k) {
            seen[nextX][nextY] = true;
            if (checkDfs(k, nextX, nextY, seen)) {
              return true;
            }
          }
        }
      }

      return false;
    }

    let l = 0, r = max - min;

    while (l < r) {
      let m = Math.floor((l + r) / 2);
      if (check(m)) {
        r = m;
      } else {
        l = m + 1;
      }
    }

    return l;
  };

  minimumEffortPathBfs(heights: number[][]): number {
    const m = heights.length;
    const n = heights[0].length;
    let max = 0, min = Infinity;
    for (let i = 0; i < m; i++) {
      max = Math.max(max, ...heights[i]);
      min = Math.min(min, ...heights[i]);
    }
    const isValid = (x: number, y: number) => {
      return x >= 0 && y >= 0 && x < m && y < n;
    };
    const directions = [[0, 1], [0, -1], [-1, 0], [1, 0]];

    const check = (k: number): boolean => {
      const seen = [];
      for (let i = 0; i < m; i++) {
        seen.push(new Array(n).fill(false));
      }
      let queue = [[0, 0]];
      seen[0][0] = true;
      while (queue.length > 0) {
        let nextQueue = [];
        for (let i = 0; i < queue.length; i++) {
          let [x, y] = queue[i];
          if (x == m - 1 && y == n - 1) {
            return true;
          }
          for (const [a, b] of directions) {
            let [nextX, nextY] = [x + a, y + b];
            if (isValid(nextX, nextY) && !seen[nextX][nextY]) {
              let diff = Math.abs(heights[x][y] - heights[nextX][nextY]);
              if (diff <= k) {
                seen[nextX][nextY] = true;
                nextQueue.push([nextX, nextY]);
              }
            }
          }
        }
        queue = nextQueue;
      }

      return false;
    }

    let l = 0, r = max - min;

    while (l < r) {
      let m = Math.floor((l + r) / 2);
      if (check(m)) {
        r = m;
      } else {
        l = m + 1;
      }
    }

    return l;
  };

  minEatingSpeed(piles: number[], h: number): number {
    piles.sort((a, b) => a - b);
    let min = 1, max = piles[piles.length - 1], mid = 0;

    const calc = (speed: number): number => {
      return piles.reduce((prev, curr) => {
        return prev + Math.ceil(curr / speed);
      }, 0);
    }

    while (min < max) {
      mid = Math.floor((min + max) / 2);
      let midHours = calc(mid);
      if (midHours <= h) {
        max = mid;
      } else {
        min = mid + 1;
      }
    }
    return min;
  };

  answerQueries(nums: number[], queries: number[]): number[] {
    nums.sort((a, b) => a - b);
    let prefix = [nums[0]];
    for (let i = 1; i < nums.length; i++) {
      prefix[i] = prefix[i - 1] + nums[i];
    }
    const bs = (arr: number[], t: number): number => {
      let l = 0, r = arr.length - 1;
      while (l < r) {
        let m = Math.floor((l + r) / 2);
        if (arr[m] >= t) {
          r = m;
        } else {
          l = m + 1;
        }
      }
      if (arr[l] <= t) {
        return l + 1;
      } else {
        return l;
      }
    };

    let result: number[] = [];
    for (let i = 0; i < queries.length; i++) {
      let r = bs(prefix, queries[i]);
      result.push(r);
    }
    return result;
  };

  searchInsert(nums: number[], target: number): number {
    let l = 0, r = nums.length - 1;
    if (nums[l] > target) {
      return 0;
    }
    if (nums[r] < target) {
      return r + 1;
    }
    while (l <= r) {
      const m = Math.floor((l + r) / 2);
      if (nums[m] == target) {
        return m;
      }
      if (nums[m] > target) {
        r = m - 1;
      } else {
        l = m + 1;
      }
    }
    if (nums[l] < target) {
      return l + 1;
    } else {
      return l;
    }
  };

  successfulPairs(spells: number[], potions: number[], success: number): number[] {
    potions.sort((a, b) => a - b);
    const findPotions = (s: number, pos: number[], t: number): number => {
      if (s * pos[0] > t) {
        return pos.length;
      }
      if (s * pos[pos.length - 1] < t) {
        return 0;
      }
      let l = 0, r = pos.length - 1;
      let m = -1;
      while (l < r) {
        m = Math.floor((l + r) / 2);
        if (pos[m] * s >= t) {
          r = m;
        } else {
          l = m + 1;
        }
      }
      return pos.length - l;
    }

    let result = [];
    for (let i = 0; i < spells.length; i++) {
      let r = findPotions(spells[i], potions, success);
      result.push(r);
    }

    return result;
  };

  searchMatrix(matrix: number[][], target: number): boolean {
    const m = matrix.length;
    const n = matrix[0].length;
    const bs = (arr: number[], t: number): boolean => {
      let left = 0, right = arr.length - 1;
      while (left <= right) {
        let mid = Math.floor((left + right) / 2);
        if (arr[mid] == t) {
          return true;
        }
        if (arr[mid] > t) {
          right = mid - 1;
        } else {
          left = mid + 1;
        }
      }
      return false;
    };

    for (let i = 0; i < m; i++) {
      if (matrix[i][0] > target || matrix[i][n - 1] < target) {
        continue;
      } else {
        let result = bs(matrix[i], target);
        return result;
      }
    }
    return false;
  };

  binarySearch(arr: number[], target: number): number {
    let left = 0;
    let right = arr.length;
    while (left < right) {
      let mid = Math.floor((left + right) / 2);
      // if (arr[mid] == target) {
      //   // do something
      //   console.log(mid);
      //   return mid;
      // }
      if (arr[mid] >= target) {
        right = mid;
      } else {
        left = mid + 1;
      }
    }

    // target is not in arr, but left is at the insertion point
    return left;
  }

  makeIntegerBeautiful(n: number, target: number): number {
    const getSum = (num: number): number => {
      return (num + '').split('').map(a => parseInt(a)).reduce((prev, curr) => prev + curr, 0);
    };
    if (getSum(n) <= target) {
      return 0;
    }
    let exp = 10;
    let round = (Math.floor(n / exp) + 1) * exp;
    while (getSum(round) > target) {
      exp = exp * 10;
      round = Math.ceil(n / exp) * exp;
    }
    return round - n;
  };

  minMoves(target: number, maxDoubles: number): number {
    if (maxDoubles == 0) {
      return target - 1;
    }
    let remaining = target;
    let count = 0;
    let dblCount = 0;
    while (remaining > 1) {
      if (dblCount < maxDoubles && remaining % 2 == 0) {
        remaining = remaining / 2;
        dblCount++;
      } else {
        remaining -= 1;
      }
      count++;
    }
    return count;
  };

  maximumEvenSplit(finalSum: number): number[] {
    let result: number[] = [];
    if (finalSum % 2 != 0) {
      return result;
    }
    let curr = 2;
    let remaining = finalSum;
    while (remaining >= curr) {
      result.push(curr);
      remaining -= curr;
      curr += 2;
    }
    let last = result.pop()!;
    result.push(last + remaining);
    return result;
  };

  largestPalindromic(num: string): string {
    let map = new Map();
    for (let i = 0; i < num.length; i++) {
      let n = parseInt(num[i]);
      if (!map.has(n)) {
        map.set(n, [n, 0]);
      }
      map.set(n, [n, map.get(n)[1] + 1]);
    }
    let arr = [...map.values()];
    arr.sort((a, b) => b[0] - a[0]);
    let result = [];
    let middleNum = -1;
    for (let i = 0; i < arr.length; i++) {
      let [n, c] = arr[i];
      if (c >= 2) {
        if (result.length == 0 && n == 0) {
          continue;
        }
        while (c > 1) {
          result.push(n);
          c = c - 2;
        }
      }
      if (c == 1 && middleNum < 0) {
        middleNum = n;
      }
    }
    let copy = [...result];
    copy.reverse();
    if (middleNum >= 0) {
      result.push(middleNum);
    }
    result = result.concat(copy);
    return result.join('') || '0';
  };

  maxArea(height: number[]): number {
    let area = 0;
    let i = 0, j = height.length - 1;
    while (i < j) {
      let lower = height[i] > height[j] ? height[j] : height[i];
      area = Math.max(area, lower * (j - i));
      if (height[i] > height[j]) {
        j--;
      } else if (height[i] < height[j]) {
        i++;
      } else {
        i++;
        j--;
      }
    }

    return area;
  };

  partitionString(s: string): number {
    let set = new Set();
    let count = 1;
    for (let i = 0; i < s.length; i++) {
      if (!set.has(s[i])) {
        set.add(s[i]);
      } else {
        count++;
        set.clear();
        set.add(s[i]);
      }
    }
    return count;
  };

  appendCharacters(s: string, t: string): number {
    let i = 0, j = 0;
    while (i < s.length && j < t.length) {
      if (s[i] == t[j]) {
        i++;
        j++;
      } else {
        i++;
      }
    }
    return t.length - j;
  };

  getSmallestString(n: number, k: number): string {
    const alphabet = ['', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    let sum = 0
    let s = ''

    for (let i = n - 1; i >= 0; i--) {
      let target = (k - i) - sum;
      if (target > 26) target = 26;
      s = alphabet[target] + s;
      sum += target;
    }

    return s;
  };

  matchPlayersAndTrainers(players: number[], trainers: number[]): number {
    players.sort((a, b) => b - a);
    trainers.sort((a, b) => b - a);
    let j = 0;
    let count = 0;
    for (let i = 0; i < players.length; i++) {
      if (j == trainers.length) {
        break;
      }
      let p = players[i];
      let t = trainers[j];
      if (p <= t) {
        count++;
        j++;
      }
    }
    return count;
  };

  largestSumAfterKNegations(nums: number[], k: number): number {
    const heap = new MinPriorityQueue<number>();
    let sum = 0;
    for (let i = 0; i < nums.length; i++) {
      heap.enqueue(nums[i]);
    }
    while (k > 0) {
      k--;
      let num = heap.dequeue()!;
      if (num < 0) {
        heap.enqueue(-num);
      } else if (num == 0) {
        k = 0;
      } else {
        heap.enqueue(-num);
      }
    }

    while (heap.size() > 0) {
      let num = heap.dequeue()!;
      sum += num;
    }

    return sum;
  };

  argestSumAfterKNegations(nums: number[], k: number): number {
    nums.sort((a, b) => a - b);
    let sum = 0;
    for (let i = 0; i < nums.length; i++) {
      if (k > 0) {
        nums[i] = -nums[i];
        k--;
        if (nums[i] == 0) {
          k = 0;
        } else if (nums[i] > 0) {
          i--;
        }
      }
      sum += nums[i];
    }

    return sum;
  };

  minSetSize(arr: number[]): number {
    let map = new Map();
    let n = arr.length;
    for (let i = 0; i < n; i++) {
      const val = arr[i];
      if (!map.has(val)) {
        map.set(val, 0);
      }
      map.set(val, map.get(val) + 1);
    }

    let values = [];
    for (const [val, occur] of map) {
      values.push([val, occur]);
    }
    values.sort((a, b) => b[1] - a[1]);

    let half = n / 2;
    let count = 0;
    for (let i = 0; i < values.length; i++) {
      let [val, occur] = values[i];
      n = n - occur;
      count++;
      if (n <= half) {
        return count;
      }
    }

    return count;
  }

  maxNumberOfApples(weight: number[]): number {
    weight.sort((a, b) => a - b);
    let sum = 0;
    let i = 0;
    for (i = 0; i < weight.length; i++) {
      sum += weight[i];
      if (sum > 5000) {
        break;
      }
    }
    return i;
  }

  maximumUnits(boxTypes: number[][], truckSize: number): number {
    boxTypes.sort((a, b) => b[1] - a[1]);
    let result = 0;
    for (let i = 0; i < boxTypes.length; i++) {
      let [numOfBoxes, numOfUnits] = boxTypes[i];
      if (numOfBoxes > truckSize) {
        result += truckSize * numOfUnits;
        break;
      } else {
        truckSize -= numOfBoxes;
        result += numOfBoxes * numOfUnits;
      }
    }

    return result;
  }

  numRescueBoats(people: number[], limit: number): number {
    people.sort((a, b) => a - b);
    let steps = 0;
    let i = 0, j = people.length - 1;
    while (i <= j) {
      let h = people[j];
      let l = people[i];
      steps++;
      if (i == j) {
        break;
      } else if (h + l <= limit) {
        j--;
        i++;
      } else {
        j--;
      }
    }
    return steps;
  };

  findLeastNumOfUniqueInts(arr: number[], k: number): number {
    const heap = new MinPriorityQueue<{ num: number, occur: number }>((p) => p.occur);
    // const heap = new MinPriorityQueue({priority: (p) => p.occur});
    const map = new Map();
    for (const n of arr) {
      if (!map.has(n)) {
        map.set(n, 0);
      }
      map.set(n, map.get(n) + 1);
    }
    for (const [num, occur] of map) {
      heap.enqueue({num, occur});
    }

    for (let i = 0; i < k; i++) {
      let {num, occur} = heap.dequeue();
      if (occur > 1) {
        occur--;
        heap.enqueue({num, occur});
      }
    }

    return heap.size();
  };

  findMaximizedCapital(k: number, w: number, profits: number[], capital: number[]): number {
    const n = profits.length;
    const projects: { capt: number, profit: number }[] = [];
    for (let i = 0; i < n; i++) {
      projects.push({capt: capital[i], profit: profits[i]});
    }
    projects.sort((a, b) => a.capt - b.capt);
    // const heap = new MaxPriorityQueue({ priority: (bid) => bid.value });
    const heap = new MaxPriorityQueue<{ index: number, profit: number }>((p) => p.profit);

    let curr = 0;
    for (let i = 0; i < k; i++) {
      while (curr < n && projects[curr].capt <= w) {
        heap.enqueue({index: curr, profit: projects[curr].profit});//add profits to max heap
        curr++;
      }
      if (heap.size() == 0) {
        return w;
      }

      w += heap.dequeue().profit!;
    }
    return w;
  }

  partitionArray(nums: number[], k: number): number {
    nums.sort((a, b) => a - b);

    let min = nums[0];
    let count = 1;
    for (let i = 1; i < nums.length; i++) {
      if (nums[i] - min > k) {
        min = nums[i];
        count++;
      }
    }

    return count;
  };

  smallestRange(nums: number[][]): number[] {
    let rangeStart: number = Number.MIN_SAFE_INTEGER;
    let rangeEnd: number = Number.MAX_SAFE_INTEGER;
    let maxVal = Number.MIN_SAFE_INTEGER;
    let minHeap = new MinPriorityQueue<{ value: number, listIndex: number, elmIndex: number }>((item => item.value));
    // Insert the first element from each list into the min-heap
    for (let i = 0; i < nums.length; i++) {
      minHeap.enqueue({value: nums[i][0], listIndex: i, elmIndex: 0});
      maxVal = Math.max(maxVal, nums[i][0]); // max from first elements
    }

    // Continue until we can't proceed further
    while (minHeap.size() == nums.length) {
      let {value: minVal, listIndex: row, elmIndex: col} = minHeap.dequeue();

      // Update the smallest range
      if (maxVal - minVal < rangeEnd - rangeStart) {
        rangeStart = minVal;
        rangeEnd = maxVal;
      }

      // If possible, add the next element from the same row to the heap
      if (col + 1 < nums[row].length) {
        let nextVal = nums[row][col + 1];
        minHeap.enqueue({value: nextVal, listIndex: row, elmIndex: col + 1});
        maxVal = Math.max(maxVal, nextVal);
      }
    }

    return [rangeStart, rangeEnd];
  };

  mostBooked(n: number, meetings: number[][]): number {
    let usedRooms = new PriorityQueue<{ end: number, room: number }>((e1, e2) => {
      if (e1.end < e2.end) return -1; // do not swap
      if (e1.end > e2.end) return 1; // swap

      // salaries are the same, compare rank
      return e1.room < e2.room ? -1 : 1;
    });
    let availableRooms = new MinPriorityQueue<{ room: number }>((item => item.room));
    let meetingCount = new Array(n).fill(0);
    let map = new Map();

    meetings.sort((a, b) => a[0] - b[0]);

    for (let i = 0; i < n; i++) {
      map.set(i, 0);
      availableRooms.enqueue({room: i});
    }

    for (const [start, end] of meetings) {
      while (!usedRooms.isEmpty() && usedRooms.front().end <= start) {
        let room = usedRooms.dequeue().room;
        availableRooms.enqueue({room});
      }

      if (!availableRooms.isEmpty()) { //if rooms available
        const room = availableRooms.dequeue().room;
        usedRooms.enqueue({end, room});
        meetingCount[room]++;
      } else { //no room available
        let firstUsed = usedRooms.dequeue();
        let roomAvailabilityTime = firstUsed.end;
        let room = firstUsed.room;
        //append the time to the existing meeting on queue
        usedRooms.enqueue({end: (roomAvailabilityTime + end - start), room});
        meetingCount[room]++;
      }
    }
    console.log('meetingCount', meetingCount);

    let maxMeetingCount = 0, maxMeetingCountRoom = 0;
    for (let i = 0; i < n; i++) {
      if (meetingCount[i] > maxMeetingCount) {
        maxMeetingCount = meetingCount[i];
        maxMeetingCountRoom = i;
      }
    }

    return maxMeetingCountRoom;
  };

  kSmallestPairs(nums1: number[], nums2: number[], k: number): number[][] {
    let m = nums1.length;
    let n = nums2.length;
    let size = k;
    let heap = new MinPriorityQueue<{ i: number, j: number, sum: number }>((item => item.sum));
    let visited = new Set();
    let i = 0, j = 0;
    heap.enqueue({i: i, j: j, sum: nums1[i] + nums2[j]});
    visited.add(`${i}-${j}`);

    const result = [];
    while (heap.size() > 0 && size > 0) {
      size--;
      let {i, j, sum} = heap.dequeue();
      result.push([nums1[i], nums2[j]]);

      if (i + 1 < m && !visited.has(`${i + 1}-${j}`)) {
        heap.enqueue({i: i + 1, j, sum: nums1[i + 1] + nums2[j]});
        visited.add(`${i + 1}-${j}`);
      }

      if (j + 1 < n && !visited.has(`${i}-${j + 1}`)) {
        heap.enqueue({i, j: j + 1, sum: nums1[i] + nums2[j + 1]});
        visited.add(`${i}-${j + 1}`);
      }
    }

    return result;
  };

  totalCost(costs: number[], k: number, candidates: number): number {
    const headWorkers = new PriorityQueue<{ index: number, cost: number }>((e1, e2): number => {
      if (e1.cost > e2.cost) return 1; // do not swap
      if (e1.cost < e2.cost) return -1; // swap
      return e1.index < e2.index ? -1 : 1;
    });
    const tailWorkers = new PriorityQueue<{ index: number, cost: number }>((e1, e2): number => {
      if (e1.cost > e2.cost) return 1; // do not swap
      if (e1.cost < e2.cost) return -1; // swap
      return e1.index < e2.index ? -1 : 1;
    });

    for (let i = 0; i < candidates; i++) {
      headWorkers.enqueue({index: i, cost: costs[i]});
    }
    for (let i = Math.max(candidates, costs.length - candidates); i < costs.length; i++) {
      tailWorkers.enqueue({index: i, cost: costs[i]});
    }
    let answer = 0;
    let nextHead = candidates;
    let nextTail = costs.length - 1 - candidates;

    for (let i = 0; i < k; i++) {
      if (tailWorkers.isEmpty() || !headWorkers.isEmpty() && headWorkers.front().cost <= tailWorkers.front().cost) {
        answer += headWorkers.dequeue().cost;

        // Only refill the queue if there are workers outside the two queues.
        if (nextHead <= nextTail) {
          headWorkers.enqueue({index: nextHead, cost: costs[nextHead]});
          nextHead++;
        }
      } else {
        answer += tailWorkers.dequeue().cost;

        // Only refill the queue if there are workers outside the two queues.
        if (nextHead <= nextTail) {
          tailWorkers.enqueue({index: nextTail, cost: costs[nextTail]});
          nextTail--;
        }
      }
    }

    return answer;
  };

  repeatLimitedString(s: string, repeatLimit: number): string {
    let queue = new MaxPriorityQueue<{ letter: string, count: number }>((item => item.letter));
    const map = new Map();
    for (let i = 0; i < s.length; i++) {
      if (!map.has(s[i])) {
        map.set(s[i], 0);
      }
      map.set(s[i], map.get(s[i]) + 1);
    }
    for (const [k, v] of map) {
      queue.enqueue({letter: k, count: v});
    }

    const result: string[] = [];
    while (queue.size() > 0) {
      let item = queue.dequeue()!;
      for (let i = 0; i < repeatLimit && item.count > 0; i++, item.count--) {
        result.push(item.letter);
      }
      if (item.count > 0) {
        if (!queue.isEmpty()) {
          const front = queue.front();
          result.push(front.letter);
          front.count--;
          if (front.count == 0) {
            queue.dequeue();
          }
          queue.enqueue(item);
        }
      }
    }

    return result.join('');
  };

  topKFrequent(words: string[], k: number): string[] {
    const map = new Map();
    for (const w of words) {
      if (!map.has(w)) {
        map.set(w, 0);
      }
      map.set(w, map.get(w) + 1);
    }

    interface Frequency {
      count: number;
      value: string;
    }

    const heap = new PriorityQueue<Frequency>(
      (e1: Frequency, e2: Frequency): number => {
        if (e1.count > e2.count) return -1; // do not swap
        if (e1.count < e2.count) return 1; // swap

        // salaries are the same, compare rank
        return e1.value > e2.value ? 1 : -1;
      }
    );
    for (const [key, value] of map) {
      heap.enqueue({count: value, value: key});
    }

    let result = [];
    for (let i = 0; i < k; i++) {
      result.push(heap.dequeue().value);
    }
    return result;
  };

  findClosestElements(arr: number[], k: number, x: number): number[] {
    interface Frequency {
      index: number;
      value: number;
    }

    const compare: ICompare<Frequency> = (a: Frequency, b: Frequency) => {
      if (Math.abs(a.value - x) < Math.abs(b.value - x)) {
        return -1;
      } else if (Math.abs(a.value - x) > Math.abs(b.value - x)) {
        // prioritize newest cars
        return 1;
      }
      return a > b ? 1 : -1;
    };
    const heap = new PriorityQueue<Frequency>(compare);
    // const heap = new PriorityQueue<Frequency>({
    //   compare: (a: Frequency, b: Frequency) => {
    //       if (Math.abs(a.value - x) > Math.abs(b.value - x)) {
    //         return -1;
    //       } else if (Math.abs(a.value - x) < Math.abs(b.value - x)) {
    //         // prioritize newest cars
    //         return 1;
    //       }
    //       return a < b ? -1 : 1;
    //     }
    // });
    for (let i = 0; i < arr.length; i++) {
      heap.enqueue({index: i, value: arr[i]});
    }

    const result = [];
    for (let i = 0; i < k; i++) {
      result.push(heap.dequeue().value);
    }

    return result.sort();
  };

  topKFrequent1(nums: number[], k: number): number[] {
    const map = new Map();
    for (let i = 0; i < nums.length; i++) {
      if (!map.has(nums[i])) {
        map.set(nums[i], 0);
      }
      map.set(nums[i], map.get(nums[i]) + 1);
    }

    interface Frequency {
      index: number;
      value: number;
    }

    const getFrequencyValue: IGetCompareValue<Frequency> = (f) => f.value;
    // const heap = new MaxPriorityQueue<Frequency>({
    //   priority: (f: Frequency) => f.value
    // });
    const heap = new MaxPriorityQueue(getFrequencyValue);

    for (let [key, value] of map) {
      heap.enqueue({index: key, value});
    }
    const result = [];
    for (let i = 0; i < k; i++) {
      result.push(heap.dequeue().index);
    }

    return result;
  }

  medianSlidingWindow(nums: number[], k: number): number[] {
    if (k == 1) {
      return nums;
    }
    const hi = new MinPriorityQueue<number>();
    const lo = new MaxPriorityQueue<number>();
    const map = new Map<number, number>();

    for (let i = 0; i < k; i++) {
      lo.enqueue(nums[i]);
    }
    for (let i = 0; i < Math.floor(k / 2); i++) {
      hi.enqueue(lo.dequeue());
    }

    let i = k;
    let result = [];

    while (true) {
      //initialization
      let med = k % 2 == 1 ? lo.front() : (lo.front() + hi.front()) / 2;
      result.push(med);

      if (i >= nums.length) {
        break;
      }

      let balance = 0;
      let out_num = nums[i - k];
      balance += out_num <= lo.front() ? -1 : 1;
      let in_num = nums[i++];

      let occur = map.get(out_num) || 0;
      map.set(out_num, occur + 1);

      if (!lo.isEmpty() && in_num <= lo.front()) {
        balance++;
        lo.enqueue(in_num);
      } else {
        balance--;
        hi.enqueue(in_num);
      }

      if (balance > 0) {
        hi.enqueue(lo.dequeue());
        balance--;
      }

      if (balance < 0) {
        lo.enqueue(hi.dequeue());
        balance++;
      }

      //remove invalid numbers tha should be discarded from heap tops
      while (!lo.isEmpty() && map.has(lo.front())) {
        let occur = map.get(lo.front())!;
        if (occur > 0) {
          map.set(lo.front(), occur - 1);
          lo.dequeue();
        } else {
          break;
        }
      }

      while (!hi.isEmpty() && map.has(hi.front())) {
        let occur = map.get(hi.front())!;
        if (occur > 0) {
          map.set(hi.front(), occur - 1);
          hi.dequeue();
        } else {
          break;
        }
      }
    }

    return result;
  };

  halveArray(nums: number[]): number {
    const heap = new MaxHeap();
    let sum = 0;
    for (const num of nums) {
      sum += num;
      heap.add(num);
    }

    let half = sum / 2;
    let operations = 0;
    while (sum > half) {
      let item = heap.remove()!;
      sum -= item / 2;
      heap.add(item / 2);
      operations++;
    }

    return operations;
  };

  lastStoneWeight(stones: number[]): number {
    const heap = new MaxPriorityQueue();
    for (const s of stones) {
      heap.enqueue(s);
    }
    while (heap.size() > 0) {
      let x = heap.dequeue();
      let y = heap.dequeue();
      if (y == null) {
        return x as number;
      }
      if (x != y) {
        heap.enqueue(Math.abs((x as number) - (y as number)));
      }
    }
    return 0;
  };

  orangesRotting(grid: number[][]): number {
    const start = [];
    const m = grid.length;
    const n = grid[0].length;
    const seen = [];
    for (let i = 0; i < m; i++) {
      seen.push(new Array(n).fill(false));
    }

    const directions = [[0, 1], [0, -1], [1, 0], [-1, 0]];
    const isValid = (i: number, j: number): boolean => {
      return i >= 0 && j >= 0 && i < m && j < n && grid[i][j] == 1;
    };
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        const ro = grid[i][j];
        if (ro == 2) {
          start.push([i, j]);
          seen[i][j] = true;
        }
      }
    }
    let time = 0;
    let queue = start;
    while (queue.length > 0) {
      const nextQueue: number[][] = [];
      for (let i = 0; i < queue.length; i++) {
        const [x, y] = queue[i];
        for (const [x1, y1] of directions) {
          let nextX = x + x1;
          let nextY = y + y1;
          if (isValid(nextX, nextY) && !seen[nextX][nextY]) {
            seen[nextX][nextY] = true;
            nextQueue.push([nextX, nextY]);
          }
        }
      }
      queue = nextQueue;
      if (queue.length > 0) {
        time++;
      }
    }

    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        if (!seen[i][j] && grid[i][j] == 1) {
          return -1;
        }
      }
    }
    return time;
  };

  equationsPossible(equations: string[]): boolean {
    let map = new Map();
    for (let i = 0; i < equations.length; i++) {
      const str = equations[i];
      if (str.indexOf('==') >= 0) {
        let [x, y] = str.split('==');
        if (!map.has(x)) {
          map.set(x, new Set());
        }
        map.get(x).add(y);
        if (!map.has(y)) {
          map.set(y, new Set());
        }
        map.get(y).add(x);
      }
    }

    const bfs = (key: string, target: string): boolean => {
      const seen = new Set();
      let queue = [key];
      seen.add(key);
      while (queue.length > 0) {
        let nextQueue = [];
        for (let i = 0; i < queue.length; i++) {
          let k = queue[i];
          if (!map.has(k)) {
            continue;
          }
          let values = map.get(k)!.keys();
          for (const v of values) {
            if (!seen.has(v)) {
              seen.add(v);
              if (v == target) {
                return false;
              }
              nextQueue.push(v);
            }
          }
        }
        queue = nextQueue;
      }
      return true;
    }

    for (let i = 0; i < equations.length; i++) {
      const str = equations[i];
      if (str.indexOf('!=') >= 0) {
        let [x, y] = str.split('!=');
        if (!bfs(x, y) || x == y) {
          return false;
        }
      }
    }

    return true;
  };

  getAncestors(n: number, edges: number[][]): number[][] {
    const map = new Map();
    for (const [x, y] of edges) {
      if (!map.has(y)) {
        map.set(y, []);
      }
      map.get(y).push(x);
    }

    const bfs = (num: number): number[] => {
      let list: number[] = [];
      let seen = new Set();
      let queue = [num];
      while (queue.length > 0) {
        let nextQueue = [];
        for (let i = 0; i < queue.length; i++) {
          let child = queue[i];
          let parents = map.get(child) || [];
          for (const p of parents) {
            if (!seen.has(p)) {
              seen.add(p);
              list.push(p);
              nextQueue.push(p);
            }
          }
        }
        queue = nextQueue;
      }

      return list.sort((a, b) => a - b);
    };

    let result = [];
    for (let i = 0; i < n; i++) {
      result.push(bfs(i));
    }
    return result;
  };

  numOfMinutes1(n: number, headID: number, manager: number[], informTime: number[]): number {
    const timeFromEmployeeToHead = (id: number): number => {
      if (manager[id] != -1) {
        informTime[id] += timeFromEmployeeToHead(manager[id]);
        manager[id] = -1;
      }
      return informTime[id];
    }

    manager.forEach((_, index) => timeFromEmployeeToHead(index));

    return Math.max(...informTime);
  };

  numOfMinutes(n: number, headID: number, manager: number[], informTime: number[]): number {
    const map = new Map();
    for (let i = 0; i < manager.length; i++) {
      let m = manager[i];
      if (m == -1) continue;
      if (!map.has(m)) {
        map.set(m, []);
      }
      map.get(m).push(i);
    }
    let queue: number[][] = [[headID, 0]];
    let maxTime = 0;
    while (queue.length > 0) {
      let nextQueue: number[][] = [];
      for (let i = 0; i < queue.length; i++) {
        let [manId, time] = queue[i];
        maxTime = Math.max(maxTime, time);
        let subs = map.get(manId) || [];
        for (const s of subs) {
          nextQueue.push([s, time + informTime[manId]]);
        }
      }
      queue = nextQueue;
    }
    return maxTime;
  };

  numEnclaves(grid: number[][]): number {
    const m = grid.length;
    const n = grid[0].length;
    const seen: boolean[][] = [];
    for (let i = 0; i < m; i++) {
      seen.push(new Array(n).fill(false));
    }
    const isValid = (x: number, y: number) => {
      return x >= 0 && y >= 0 && x < m && y < n && grid[x][y] == 1;
    }

    const isEdge = (x: number, y: number) => {
      return (x == 0 || y == 0 || x == m - 1 || y == n - 1) && grid[x][y] == 1;
    }

    const directions = [[0, 1], [0, -1], [1, 0], [-1, 0]];

    const bfs = (start: number[]): number => {
      let queue: number[][] = [start];
      seen[start[0]][start[1]] = true;
      let count = 1;
      let isEdgeConnected = false;
      while (queue.length > 0) {
        let nextQueue: number[][] = [];
        for (let i = 0; i < queue.length; i++) {
          let [x, y] = queue[i];
          if (isEdge(x, y)) {
            isEdgeConnected = true;
          }

          for (const [x1, y1] of directions) {
            let nextX = x + x1;
            let nextY = y + y1;
            if (isValid(nextX, nextY) && !seen[nextX][nextY]) {
              seen[nextX][nextY] = true;
              nextQueue.push([nextX, nextY]);
            }
          }
        }
        count += nextQueue.length;
        queue = nextQueue;
      }
      return isEdgeConnected ? 0 : count;
    };

    let count = 0;
    //find start cell
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        if (grid[i][j] == 1 && !seen[i][j]) {
          count += bfs([i, j]);
        }
      }
    }

    return count;
  };

  islandPerimeter(grid: number[][]): number {
    const m = grid.length;
    const n = grid[0].length;

    const isValid = (x: number, y: number) => {
      return x >= 0 && y >= 0 && x < m && y < n && grid[x][y] == 1;
    }
    const directions = [[0, 1], [0, -1], [1, 0], [-1, 0]];

    const countNeighbors = (x: number, y: number): number => {
      let count = 0;
      for (const [x1, y1] of directions) {
        let nextX = x + x1;
        let nextY = y + y1;
        if (isValid(nextX, nextY)) {
          count++;
        }
      }
      return count;
    }

    let perimeter = 0;
    //find start cell
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        if (grid[i][j] == 1) {
          let neighbors = countNeighbors(i, j);
          perimeter += 4 - neighbors;
        }
      }
    }

    return perimeter;
  };

  maximalNetworkRank(n: number, roads: number[][]): number {
    const map = new Map();
    const set = new Set();
    for (let i = 0; i < roads.length; i++) {
      let [a, b] = roads[i];
      set.add(`${a}_${b}`);
      set.add(`${b}_${a}`);
      if (!map.has(a)) {
        map.set(a, []);
      }
      map.get(a).push(b);
      if (!map.has(b)) {
        map.set(b, []);
      }
      map.get(b).push(a);
    }
    let max = 0;
    let result = [];
    for (let i = 0; i < n; i++) {
      let arr = map.get(i) || [];
      result.push([i, arr.length]);
    }
    result.sort((a: number[], b: number[]) => a[1] - b[1]);
    let size = result.length;

    let j = size - 2;
    while (j >= 0) {
      for (let i = size - 1; i > 0; i--) {
        if (i == j) {
          continue;
        }
        let sum = result[i][1] + result[j][1];
        let a = result[i][0], b = result[j][0];
        if (set.has(`${a}_${b}`) || set.has(`${b}_${a}`)) {
          sum = sum - 1;
        }
        max = Math.max(sum, max);
      }
      j--;
    }

    return max;
  };

  findJudge(n: number, trust: number[][]): number {
    if (trust.length == 0) {
      return n == 1 ? 1 : -1;
    }
    let map = new Map();
    let set = new Set();
    for (const [a, b] of trust) {
      set.add(a);
      if (!map.has(b)) {
        map.set(b, []);
      }
      map.get(b).push(a);
    }
    for (const key of map.keys()) {
      if (!set.has(key) && map.get(key).length == n - 1) {
        return key;
      }
    }

    return -1;
  };

  deleteNode(root: TreeNode | null, key: number): TreeNode | null {
    if (!root) {
      return null;
    }
    const getParent = (node: TreeNode | null, parent: TreeNode | null): (TreeNode | null)[] => {
      if (!node) {
        return [];
      }
      if (node.val == key) {
        return [node, parent];
      }
      let result = [];
      if (node.val > key) {
        result = getParent(node.left, node);
      } else {
        result = getParent(node.right, node);
      }
      return result;
    };

    const findReplace = (node: TreeNode | null): TreeNode | null => {
      if (!node) {
        return null;
      }
      if (node.right) {
        node = node.right;
        while (node.left) {
          node = node.left;
        }
        return node;
      }
      return null;
    };

    let [target, parent] = getParent(root, null);

    if (!target) {//not found
      return root;
    }
    let replaceNode = findReplace(target) || parent;

    replaceNode!.left = target!.left;

    return root;
  };

  lowestCommonAncestor(root: TreeNode | null, p: TreeNode | null, q: TreeNode | null): TreeNode | null {
    if (!root) {
      return null;
    }

    let node = null;
    if (root.val > p!.val && root.val > q!.val) {
      node = this.lowestCommonAncestor(root.left, p, q);
    } else if (root.val < p!.val && root.val < q!.val) {
      node = this.lowestCommonAncestor(root.right, p, q);
    } else {
      node = root;
    }
    return node;
  };

  getAllElements(root1: TreeNode | null, root2: TreeNode | null): number[] {
    let result1: number[] = [];
    let result2: number[] = [];

    const dfs = (node: TreeNode | null, result: number[]) => {
      if (!node) {
        return;
      }

      dfs(node.left, result);
      result.push(node.val);
      dfs(node.right, result);
    };

    dfs(root1, result1);
    dfs(root2, result2);

    let result: number[] = [];
    let j = 0;
    let i = 0;
    while (i < result1.length && j < result2.length) {
      if (result1[i] > result2[j]) {
        result.push(result2[j]);
        j++;
      } else {
        result.push(result1[i]);
        i++;
      }
    }
    for (let index = i; index < result1.length; index++) {
      result.push(result1[index]);
    }
    for (let index = j; index < result2.length; index++) {
      result.push(result2[index]);
    }

    return result;
  };

  isEvenOddTree(root: TreeNode | null): boolean {
    if (!root || root.val % 2 == 0) {
      return false;
    }

    let level = 0
    let queue = [root];
    while (queue.length > 0) {
      let nextQueue: TreeNode[] = [];
      let temp = [];
      for (let i = 0; i < queue.length; i++) {
        let node = queue[i];
        if (level % 2 == 0 && node.val % 2 == 0) { //even row, check increasing
          return false;
        }
        if (level % 2 == 1 && node.val % 2 == 1) {//odd row, check decreasing
          return false;
        }
        if (temp.length > 0) {
          if (level % 2 == 0) { //even row, check increasing
            if (node.val <= temp[temp.length - 1]) {
              return false;
            }
          } else { //odd row, check decreasing
            if (node.val >= temp[temp.length - 1]) {
              return false;
            }
          }
        }
        temp.push(node!.val);

        if (node!.left) {
          nextQueue.push(node!.left);
        }
        if (node!.right) {
          nextQueue.push(node!.right);
        }
      }
      level++;
      queue = nextQueue;
    }
    return true;
  };

  levelOrder(root: TreeNode | null): number[][] {
    if (!root) {
      return [];
    }
    let result: number[][] = [];

    let queue = [root];
    while (queue.length > 0) {
      let nextQueue: TreeNode[] = [];
      let temp = [];
      for (let i = 0; i < queue.length; i++) {
        let node = queue[i];
        temp.push(node!.val);
        if (node!.left) {
          nextQueue.push(node!.left);
        }
        if (node!.right) {
          nextQueue.push(node!.right);
        }
      }
      result.push(temp);
      queue = nextQueue;
    }
    return result;
  };

  longestZigZag(root: TreeNode | null): number {
    if (!root || (!root.left && !root.right)) {
      return 0;
    }

    const map = new Map();

    const LEFT = 0;
    const RIGHT = 1;

    let max = 0;
    const dfs = (node: TreeNode | null, nextDir: number): number => {
      if (!node) {
        return 0;
      }

      if (map.has(node)) {
        let val = map.get(node)[nextDir];
        if (val >= 0) {
          return val;
        }
      }

      let count = 0;
      if (nextDir == LEFT && node.left) {
        count++;
        count += dfs(node.left, RIGHT);
      }
      if (nextDir == RIGHT && node.right) {
        count++;
        count += dfs(node.right, LEFT);
      }


      if (!map.has(node)) {
        map.set(node, [-1, -1]);
      }
      map.get(node)[nextDir] = count;
      return count;
    };

    const travel = (node: TreeNode | null) => {
      if (!node) {
        return;
      }

      let right = dfs(node, RIGHT);
      let left = dfs(node, LEFT);
      max = Math.max(max, left, right);

      travel(node.left);
      travel(node.right);
    };

    travel(root);

    return max;
  };

  pathSum4(root: TreeNode | null, targetSum: number): number {
    const map = new Map();
    //process prefix sum
    let count = 0;
    const dfs = (node: TreeNode | null, currSum: number) => {
      if (!node) {
        return;
      }
      currSum += node.val;
      if (currSum == targetSum) {
        count++;
      }

      count += map.get(currSum - targetSum) || 0;

      if (!map.has(currSum)) {
        map.set(currSum, 0);
      }
      map.set(currSum, map.get(currSum) + 1);

      dfs(node.left, currSum);
      dfs(node.right, currSum);

      map.set(currSum, map.get(currSum) - 1)
    }

    dfs(root, 0);

    return count;
  }

  pathSum3(root: TreeNode | null, targetSum: number): number {
    const seen = new Set();

    let nodeList: TreeNode[] = [];

    const travel = (node: TreeNode | null) => {
      if (!node) {
        return;
      }

      nodeList.push(node);

      travel(node.left);
      travel(node.right);
    }

    const dfs = (node: TreeNode | null, sum: number): number => {
      let count = 0;

      if (!node) {
        return 0;
      }

      sum += node.val;

      if (sum == targetSum) {
        count++;
      }

      count += dfs(node.left, sum);
      count += dfs(node.right, sum);

      return count;
    };

    travel(root);

    let result = 0;
    for (const node of nodeList) {
      result += dfs(node, 0);
    }

    return result;
  };

  removeLeafNodes(root: TreeNode | null, target: number): TreeNode | null {
    if (!root) {
      return null;
    }

    const dfs = (node: TreeNode | null): TreeNode | null => {
      if (!node) {
        return null;
      }

      node.left = dfs(node.left);
      node.right = dfs(node.right);

      if (node.val == target && !node.left && !node.right) {
        return null;
      }

      return node;
    }

    root = dfs(root);

    return root;
  };

  pathSum(root: TreeNode | null, targetSum: number): number[][] {
    let result: number[][] = [];
    const dfs = (node: TreeNode | null, list: number[], sum: number) => {
      if (!node) {
        return;
      }

      list.push(node.val);
      sum += node.val;

      // if(sum > targetSum) {
      //   return;
      // }

      if (!node.left && !node.right && sum == targetSum) {
        result.push([...list]);
        return;
      }

      dfs(node.left, [...list], sum);
      dfs(node.right, [...list], sum);


    };

    dfs(root, [], 0);

    return result;
  };

  isSymmetric(root: TreeNode | null): boolean {
    if (!root) {
      return true;
    }

    const isMirror = (node1: TreeNode | null, node2: TreeNode | null): boolean => {
      if (!node1 && !node2) {
        return true;
      }
      if (!node1 || !node2) {
        return false;
      }

      let leftSame = isMirror(node1!.left, node2!.right);
      let rightSame = isMirror(node1!.right, node2!.left);
      let result = node1!.val == node2!.val;
      return leftSame && rightSame && result;
    }


    return isMirror(root, root);
  };

  invertTree(root: TreeNode | null): TreeNode | null {
    if (!root) {
      return null;
    }

    let temp = root.left;
    root.left = root.right;
    root.right = temp;

    this.invertTree(root.left);
    this.invertTree(root.right);

    return root;
  };

  leafSimilar(root1: TreeNode | null, root2: TreeNode | null): boolean {
    const dfs = (node: TreeNode | null): number[] => {
      let result: number[] = [];
      if (!node) {
        return result;
      }
      if (node && !node.left && !node.right) {
        result.push(node.val);
        return result;
      }
      let leftResult = dfs(node.left);
      let rightResult = dfs(node.right);
      return [...leftResult, ...rightResult];
    }
    const r1 = dfs(root1);
    const r2 = dfs(root2);
    if (r1.length !== r2.length) {
      return false;
    }

    for (let i = 0; i < r1.length; i++) {
      if (r1[i] !== r2[i]) {
        return false;
      }
    }
    return true;
  };

  isSameTree(p: TreeNode | null, q: TreeNode | null): boolean {
    if (p && q && p.val != q.val) {
      return false;
    }
    if ((p && !q) || (!p && q)) {
      return false;
    }
    if (!p && !q) {
      return true;
    }

    let isLeftSame = this.isSameTree(p!.left, q!.left)
    let isRightSame = this.isSameTree(p!.right, q!.right)

    return isLeftSame && isRightSame;

  };

  ladderLength(beginWord: string, endWord: string, wordList: string[]): number {
    const map = new Map();
    const candid = new Set();
    for (let i = 0; i < beginWord.length; i++) {
      if (!map.has(i)) {
        map.set(i, new Set());
      }
      let set = map.get(i);
      for (const word of wordList) {
        candid.add(word);
        set.add(word[i])
      }
    }
    let seen = new Set();

    let queue = [beginWord];
    let steps = 0;
    while (queue.length > 0) {
      let nextQueue = [];
      for (let i = 0; i < queue.length; i++) {
        let word = queue[i];
        if (word == endWord) {
          return steps + 1;
        }
        for (let j = 0; j < word.length; j++) {
          for (const option of map.get(j).keys()) {
            let nextWord = word.slice(0, j) + option + word.slice(j + 1);
            if (!seen.has(nextWord) && candid.has(nextWord)) {
              seen.add(nextWord);
              console.log(nextWord, steps);
              nextQueue.push(nextWord)
            }
          }
        }
      }
      steps++;
      queue = nextQueue;
    }

    return 0;
  };

  maximumDetonation(bombs: number[][]): number {
    const getKey = (x: number, y: number, r: number, i: number): string => {
      return `${x}_${y}_${r}_${i}`;
    }

    const cache = new Map();

    const calc = (x: number, y: number, r: number): number[][] => {
      const result = [];
      for (let i = 0; i < bombs.length; i++) {
        let [x1, y1, r1] = bombs[i];
        let dist = Math.sqrt((x - x1) * (x - x1) + (y - y1) * (y - y1));
        if (dist <= r) {
          result.push([x1, y1, r1, i]);
        }
      }
      return result;
    }

    const bfs = (b: number[], idx: number): number => {
      const seen = new Set<string>();
      let queue: number[][] = [b];
      let count = 1;
      seen.add(getKey(b[0], b[1], b[2], idx));
      while (queue.length > 0) {
        let nextQueue: number[][] = [];

        for (const [x, y, r, i] of queue) {
          let key = getKey(x, y, r, i);
          let result = null;
          if (cache.has(key)) {
            result = cache.get(key);
            // console.log('get value from cache', result);
          } else {
            result = calc(x, y, r);
            // console.log('calc [x, y, r] ', x, y, r);
            cache.set(key, result);
          }
          for (const [x1, y1, r1, i1] of result) {
            let key = getKey(x1, y1, r1, i1);
            if (!seen.has(key)) {
              seen.add(key);
              nextQueue.push([x1, y1, r1, i1]);
            }
          }
        }

        count += nextQueue.length;
        queue = nextQueue;
      }
      return count;
    }

    let max = 0;
    for (let i = 0; i < bombs.length; i++) {
      max = Math.max(max, bfs(bombs[i], i));
    }

    return max;
  };

  canReach(arr: number[], start: number): boolean {
    let seen = new Set();
    let n = arr.length;
    const isValid = (i: number): boolean => {
      return i >= 0 && i < n;
    }
    let queue: number[] = [start];
    seen.add(start);
    while (queue.length > 0) {
      let nextQueue: number[] = [];
      for (let i = 0; i < queue.length; i++) {
        let index = queue[i];
        let value = arr[index];
        if (value == 0) {
          return true;
        }
        for (const next of [index + value, index - value]) {
          if (isValid(next) && !seen.has(next)) {
            seen.add(next);
            nextQueue.push(next);
          }
        }
      }
      queue = nextQueue;
    }
    return false;
  };

  minMutation(startGene: string, endGene: string, bank: string[]): number {
    let map = new Set();
    for (const gene of bank) {
      map.add(gene);
    }

    let seen = new Set();

    const getNext = (s: string): string[] => {
      return ['A', 'C', 'G', 'T'].filter(a => a != s);
    }

    const bfs = (start: string, target: string): number => {
      let queue: string[] = [start];
      let steps = 0;
      seen.add(start);
      while (queue.length > 0) {
        let nextQueue: string[] = [];
        for (let i = 0; i < queue.length; i++) {
          let temp = queue[i];
          if (temp == target) {
            return steps;
          }
          let startQueue: string[] = temp.split('');
          for (let i = 0; i < startQueue.length; i++) {
            let s = startQueue[i];
            for (const nextS of getNext(s)) {
              startQueue[i] = nextS;
              let next = startQueue.join('');
              if (map.has(next) && !seen.has(next)) {
                seen.add(next);
                nextQueue.push(next);
              }
              startQueue[i] = s;
            }
          }
        }
        steps++;
        queue = nextQueue;
      }
      return -1;
    };

    return bfs(startGene, endGene);
  };

  calcEquation(equations: string[][], values: number[], queries: string[][]): number[] {
    const map = new Map();

    for (let i = 0; i < equations.length; i++) {
      let [a, b] = equations[i];
      let value = values[i];
      if (!map.has(a)) {
        map.set(a, new Map());
      }
      if (!map.has(b)) {
        map.set(b, new Map());
      }
      map.get(a).set(b, value);
      map.get(b).set(a, 1 / value);
    }

    const result = [];

    const bfs = (s: string, q: string): number => {
      let queue: { a: string, val: number }[] = [{a: s, val: 1}];
      let seen = new Set();
      seen.add(s);
      while (queue.length > 0) {
        let nextQueue: { a: string, val: number }[] = [];
        for (let i = 0; i < queue.length; i++) {
          let {a, val} = queue[i];
          if (map.has(a)) {
            let set = map.get(a);
            if (set.has(q)) {
              //cache
              map.get(s).set(q, set.get(q) * val);
              return map.get(s).get(q);
            } else {
              //cache
              for (const k of set.keys()) {
                let temp = set.get(k) * val;
                if (!map.get(s).has(k)) {
                  map.get(s).set(k, temp);
                }
                if (!seen.has(k)) {
                  seen.add(k);
                  nextQueue.push({a: k, val: temp});
                }
              }
            }
          }

        }
        queue = nextQueue;
      }

      return -1;
    }

    for (const [s, q] of queries) {
      result.push(bfs(s, q));
    }

    return result;
  };

  openLock(deadends: string[], target: string): number {
    const getNext = (arr: number[], i: number): number[][] => {
      let n = arr[i];
      let x = n - 1 >= 0 ? n - 1 : 9;
      let y = n + 1 <= 9 ? n + 1 : 0;
      let result: number[][] = [];
      arr[i] = x;
      result.push([...arr]);
      arr[i] = y;
      result.push([...arr]);
      return result;
    }

    const seen = new Set();
    const dead = new Set();
    for (const d of deadends) {
      dead.add(d);
    }

    const isValid = (arr: number[]): boolean => {
      let str = arr.join('');
      return !dead.has(str);
    }
    if (!isValid([0, 0, 0, 0])) {
      return -1;
    }
    let queue = [[0, 0, 0, 0, 0]];
    seen.add('0000');
    while (queue.length > 0) {
      let nextQueue: number[][] = [];
      for (let i = 0; i < queue.length; i++) {
        let [a, b, c, d, steps] = queue[i];
        if (`${a}${b}${c}${d}` == target) {
          return steps;
        }
        for (let i = 0; i < 4; i++) {
          for (const [a1, b1, c1, d1] of getNext([a, b, c, d], i)) {
            if (isValid([a1, b1, c1, d1]) && !seen.has(`${a1}${b1}${c1}${d1}`)) {
              seen.add(`${a1}${b1}${c1}${d1}`);
              nextQueue.push([a1, b1, c1, d1, steps + 1]);
            }
          }
        }
      }
      queue = nextQueue;
    }

    return -1;

  };

  snakesAndLadders(board: number[][]): number {
    let routes = new Map()
    const n = board.length;
    let index = n * n;
    for (let i = 0; i < n; i++) {
      let row = Math.ceil(index / n);
      if (row % 2 == 0) {
        for (let j = 0; j < n; j++) {
          routes.set(index, [i, j]);
          index--;
        }
      } else {
        for (let j = n - 1; j >= 0; j--) {
          routes.set(index, [i, j]);
          index--;
        }
      }
    }

    const seen: boolean[] = [];
    for (let i = 1; i <= n * n; i++) {
      seen.push(false);
    }

    const getNext = (curr: number, steps: number): number[][] => {
      const nextArray: number[][] = [];

      for (let i = 1; i <= 6; i++) {
        let next = curr + i;
        if (next > n * n) {
          break;
        }
        let [x, y] = routes.get(next);
        let num = board[x][y];
        if (num != -1) {//snake
          nextArray.push([num, steps]);
        } else {
          nextArray.push([next, steps]);
        }
      }
      return nextArray;
    }

    let queue = [[1, 0]];
    while (queue.length > 0) {
      let nextQueue: number[][] = [];
      for (let i = 0; i < queue.length; i++) {
        let [idx, steps] = queue[i];
        if (idx == n * n) {
          return steps;
        }

        for (const [nextNum, nextSteps] of getNext(idx, steps + 1)) {
          if (!seen[nextNum]) {
            seen[nextNum] = true;
            nextQueue.push([nextNum, nextSteps]);
          }
        }
      }
      queue = nextQueue;
    }

    return -1;
  };

  nearestExit(maze: string[][], entrance: number[]): number {
    let directions: number[][] = [[0, 1], [0, -1], [1, 0], [-1, 0]];
    const m = maze.length;
    const n = maze[0].length;
    const seen = [];
    for (let i = 0; i < m; i++) {
      seen.push(new Array(n).fill(false));
    }

    const isValid = (i: number, j: number): boolean => {
      return i < m && i >= 0 && j < n && j >= 0 && maze[i][j] == '.';
    }
    let queue: number[][] = [[entrance[0], entrance[1], 0]];
    seen[entrance[0]][entrance[1]] = true;
    while (queue.length > 0) {
      let nextQueue: number[][] = [];
      for (let i = 0; i < queue.length; i++) {
        let [x, y, steps] = queue[i];
        if (isValid(x, y) && (x == 0 || y == 0 || x == m - 1 || y == n - 1) && (x != entrance[0] || y != entrance[1])) {
          return steps;
        }

        for (const [x1, y1] of directions) {
          const nextRow = x + x1;
          const nextCol = y + y1;
          if (isValid(nextRow, nextCol) && !seen[nextRow][nextCol]) {
            seen[nextRow][nextCol] = true;
            nextQueue.push([nextRow, nextCol, steps + 1]);
          }
        }
      }
      queue = nextQueue;
    }

    return -1;
  };


  shortestAlternatingPaths(n: number, redEdges: number[][], blueEdges: number[][]): number[] {
    let addToGraph = (color: number, edges: number[][]) => {
      for (let i = 0; i < n; i++) {
        graph.get(color).set(i, []);
      }

      for (const [x, y] of edges) {
        graph.get(color).get(x).push(y);
      }
    }

    const RED = 0;
    const BLUE = 1;

    let graph = new Map();
    graph.set(RED, new Map());
    graph.set(BLUE, new Map());
    addToGraph(RED, redEdges);
    addToGraph(BLUE, blueEdges);

    let ans = new Array(n).fill(Infinity);
    let queue = [[0, RED], [0, BLUE]];
    let seen = [];
    for (let i = 0; i < n; i++) {
      seen.push(new Array(2).fill(false));
    }

    seen[0][RED] = true;
    seen[0][BLUE] = true;

    let steps = 0;

    while (queue.length) {
      let currentLength = queue.length;
      let nextQueue = [];

      for (let i = 0; i < currentLength; i++) {
        let [node, color] = queue[i];
        ans[node] = Math.min(ans[node], steps);

        for (const neighbor of graph.get(color).get(node)) {
          if (!seen[neighbor][1 - color]) {
            seen[neighbor][1 - color] = true;
            nextQueue.push([neighbor, 1 - color]);
          }
        }
      }

      queue = nextQueue;
      steps++;
    }

    for (let i = 0; i < n; i++) {
      if (ans[i] == Infinity) {
        ans[i] = -1;
      }
    }

    return ans;
  };

  getFood(grid: string[][]): number {
    const m = grid.length;
    const n = grid[0].length;
    const seen = [];
    for (let i = 0; i < m; i++) {
      seen.push(new Array(n).fill(false));
    }
    const isValid = (i: number, j: number): boolean => {
      return i < m && i >= 0 && j < n && j >= 0 && grid[i][j] != 'X';
    }
    let queue = [];
    for (let i = 0; i < m; i++) {
      if (queue.length > 0) {
        break;
      }
      for (let j = 0; j < n; j++) {
        if (grid[i][j] == '*') {
          queue.push([i, j, 0]);
          break;
        }
      }
    }
    let directions: number[][] = [[0, 1], [0, -1], [1, 0], [-1, 0]];

    while (queue.length > 0) {
      let nextQueue: number[][] = [];
      for (let [x, y, d] of queue) {
        if (grid[x][y] == '#') {
          return d;
        }

        for (const [x1, y1] of directions) {
          let nextRow = x + x1;
          let nextCol = y + y1;
          if (isValid(nextRow, nextCol) && !seen[nextRow][nextCol]) {
            seen[nextRow][nextCol] = true;
            nextQueue.push([nextRow, nextCol, d + 1]);
          }
        }
      }
      queue = nextQueue;
    }

    return -1;
  };

  shortestPath(grid: number[][], k: number): number {
    const m = grid.length;
    const n = grid[0].length;
    const seen = [];
    for (let i = 0; i < m; i++) {
      seen.push(new Array(n).fill(-1));
    }
    const isValid = (i: number, j: number): boolean => {
      return i < m && i >= 0 && j < n && j >= 0;
    }

    let directions: number[][] = [[0, 1], [0, -1], [1, 0], [-1, 0]];

    let queue = [[0, 0, k, 0]];
    while (queue.length > 0) {
      let nextQueue: number[][] = [];
      for (let [x, y, obstacles, steps] of queue) {
        if (x == m - 1 && y == n - 1) {
          return steps;
        }

        // if the current square is an obstacle, we need to spend one of our removals
        if (grid[x][y] == 1) {
          if (obstacles == 0) {
            continue;
          } else {
            obstacles--;
          }
        }
        if (seen[x][y] >= obstacles) {
          continue;
        }

        seen[x][y] = obstacles;
        for (const [x1, y1] of directions) {
          let nextRow = x + x1;
          let nextCol = y + y1;
          if (isValid(nextRow, nextCol)) {
            nextQueue.push([nextRow, nextCol, obstacles, steps + 1]);
          }
        }
      }
      queue = nextQueue;
    }

    return -1;
  };


  updateMatrix(mat: number[][]): number[][] {
    const m = mat.length;
    const n = mat[0].length;
    const seen = [];
    for (let i = 0; i < m; i++) {
      seen.push(new Array(n).fill(false));
    }
    const isValid = (i: number, j: number): boolean => {
      return i < m && i >= 0 && j < n && j >= 0 && mat[i][j] == 1;
    }

    let directions: number[][] = [[0, 1], [0, -1], [1, 0], [-1, 0]];

    let queue = [];
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        if (mat[i][j] == 0) {
          queue.push([i, j]);
          seen[i][j] = true;
        }
      }
    }

    let dist = 0;
    while (queue.length > 0) {
      let nextQueue: number[][] = [];
      dist++;
      for (const [x, y] of queue) {
        for (const [x1, y1] of directions) {
          if (isValid(x + x1, y + y1) && !seen[x + x1][y + y1]) {
            seen[x + x1][y + y1] = true;
            mat[x + x1][y + y1] = dist;
            nextQueue.push([x + x1, y + y1]);
          }
        }
      }
      queue = nextQueue;
    }

    return mat;
  };

  distanceK(root: TreeNode | null, target: TreeNode | null, k: number): number[] {
    const seen = new Set();

    const travel = (node: TreeNode | null, parent: TreeNode | null) => {
      if (!node) {
        return;
      }

      node.parent = parent;
      travel(node.left, node);
      travel(node.right, node);
    };

    travel(root, null);

    let queue: TreeNode[] = [target!];
    let level = 0;
    while (queue.length > 0) {
      let nextQueue: TreeNode[] = [];
      if (level == k) {
        return queue.map(q => q!.val);
      }
      for (const n of queue) {
        if (!seen.has(n)) {
          seen.add(target);

          for (const next of [n.left, n.right, n.parent]) {
            if (next && !seen.has(next)) {
              nextQueue.push(next!);
            }
          }
        }
      }
      level++;
      queue = nextQueue;
    }

    return [];
  };

  distanceK1(root: TreeNode | null, target: TreeNode | null, k: number): number[] {
    const map = new Map();
    const seen = new Set();

    if (k == 0) {
      return [target!.val];
    }
    const dfs = (node: TreeNode | null) => {
      if (!node) {
        return;
      }
      if (!map.has(node.val)) {
        map.set(node.val, []);
      }
      if (node.left) {
        map.get(node.val).push(node.left.val);
        if (!map.has(node.left.val)) {
          map.set(node.left.val, []);
        }
        map.get(node.left.val).push(node.val);
      }
      if (node.right) {
        map.get(node.val).push(node.right.val);
        if (!map.has(node.right.val)) {
          map.set(node.right.val, []);
        }
        map.get(node.right.val).push(node.val);
      }
      dfs(node.left);
      dfs(node.right);
    };

    dfs(root);

    let level = 0;
    let queue = map.get(target!.val);
    seen.add(target!.val);
    while (queue.length) {
      let nextQueue = [];
      level++;
      if (level == k) {
        return queue;
      }
      for (const n of queue) {
        if (!seen.has(n)) {
          if (map.has(n)) {
            for (const num of map.get(n)) {
              if (!seen.has(num)) {
                seen.add(n);
                nextQueue.push(num);
              }
            }
          }
        }
      }
      queue = nextQueue;
    }
    return [];
  };

  shortestPathBinaryMatrix(grid: number[][]): number {
    let n = grid.length;
    let seen = new Array();
    for (let i = 0; i < n; i++) {
      seen.push(new Array(n).fill(false));
    }

    let directions: number[][] = [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]];
    const isValid = (i: number, j: number): boolean => {
      return i < n && i >= 0 && j < n && j >= 0 && grid[i][j] == 0;
    }

    let min = -1;
    const bfs = (i: number, j: number) => {
      let queue = [[i, j, 0]];
      while (queue.length > 0) {
        let nextQueue = [];
        for (let idx = 0; idx < queue.length; idx++) {
          let [x, y, count] = queue[idx];
          if (!seen[x][y] && isValid(x, y)) {
            seen[x][y] = true;
            count++;
            if (x == n - 1 && y == n - 1) {
              min = count;
              break;
            }
            for (const [x1, y1] of directions) {
              let r = x1 + x;
              let c = y1 + y;
              if (isValid(r, c) && !seen[r][c]) {
                nextQueue.push([r, c, count])
              }
            }
          }
        }
        queue = nextQueue;
      }
    }

    bfs(0, 0);

    return min;
  };

  reachableNodes(n: number, edges: number[][], restricted: number[]): number {
    const map = new Map();
    let numOfRows = edges.length; // num of rows
    let seen = new Set();
    for (const r of restricted) {
      seen.add(r);
    }

    for (let i = 0; i < numOfRows; i++) {
      let [x, y] = edges[i];
      if (!map.has(x)) {
        map.set(x, []);
      }
      map.get(x).push(y);
      if (!map.has(y)) {
        map.set(y, []);
      }
      map.get(y).push(x);
    }

    const dfs = (v: number): number => {
      let count = 0;
      if (!seen.has(v)) {
        seen.add(v);
        count++;
        if (map.has(v)) {
          let neighbors = map.get(v);
          for (const nb of neighbors) {
            count += dfs(nb);
          }
        }
      }
      return count;
    };
    return dfs(0);
  };

  maxAreaOfIsland(grid: number[][]): number {
    let m = grid.length; // num of rows
    let n = grid[0].length; //num of cols
    let seen = new Array(m);
    for (let i = 0; i < m; i++) {
      seen[i] = new Array(n).fill(false);
    }

    let directions: number[][] = [[0, 1], [0, -1], [1, 0], [-1, 0]];
    const isValid = (i: number, j: number): boolean => {
      return i < m && i >= 0 && j < n && j >= 0 && grid[i][j] == 1;
    }

    const dfs = (x: number, y: number): number => {
      let area = 0;
      if (!seen[x][y] && grid[x][y] == 1) {
        seen[x][y] = true;
        area++;
      }
      for (const [x1, y1] of directions) {
        let row = x + x1;
        let col = y + y1;
        if (isValid(row, col) && !seen[row][col]) {
          area += dfs(row, col);
        }
      }
      return area;
    }

    let max = 0;
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let val = grid[i][j];
        if (val == 1 && !seen[i][j]) {
          let area = dfs(i, j);
          max = Math.max(max, area);
        }
      }
    }
    return max;
  };

  countComponents(n: number, edges: number[][]): number {
    let map = new Map();
    let seen = new Set();
    for (let i = 0; i < edges.length; i++) {
      let [x, y] = edges[i];
      if (!map.has(x)) {
        map.set(x, []);
      }
      map.get(x).push(y);

      if (!map.has(y)) {
        map.set(y, []);
      }
      map.get(y).push(x);
    }

    let count = 0;
    const dfs = (v: number) => {
      if (!seen.has(v)) {
        seen.add(v);
        if (map.has(v)) {
          let neighbors = map.get(v);
          for (const nb of neighbors) {
            dfs(nb);
          }
        }
      }
    }

    for (let i = 0; i < n; i++) {
      if (!seen.has(i)) {
        count++;
        dfs(i);
      }
    }

    return count;
  };

  validPath(n: number, edges: number[][], source: number, destination: number): boolean {
    let map = new Map();
    let seen = new Set();

    for (let i = 0; i < edges.length; i++) {
      let [x, y] = edges[i];
      if (!map.has(x)) {
        map.set(x, []);
      }
      map.get(x).push(y);

      if (!map.has(y)) {
        map.set(y, []);
      }
      map.get(y).push(x);
    }

    const dfs = (v: number): boolean => {
      if (v == destination) {
        return true;
      }
      if (!seen.has(v)) {
        seen.add(v);
        let neighbors = map.get(v);
        for (const n of neighbors) {
          if (dfs(n)) {
            return true;
          }
        }
      }
      return false;
    }

    return dfs(source);
  };

  findSmallestSetOfVertices(n: number, edges: number[][]): number[] {
    let map = new Map();

    for (let i = 0; i < edges.length; i++) {
      let [x, y] = edges[i];
      if (!map.has(y)) {
        map.set(y, []);
      }
      map.get(y).push(x);
    }
    const result = [];
    for (let i = 0; i < n; i++) {
      if (!map.has(i)) {
        result.push(i);
      }
    }
    return result;
  };

  canVisitAllRooms(rooms: number[][]): boolean {
    const seen = new Array(rooms.length).fill(false);
    seen[0] = true;
    const dfs = (k: number) => {
      let keys = rooms[k];
      for (const key of keys) {
        if (!seen[key]) {
          seen[key] = true;
          dfs(key);
        }
      }
    }
    dfs(0);
    let result = true;
    for (let i = 0; i < seen.length; i++) {
      result &&= seen[i];
    }

    return result;
  };

  minEdgeReversals(n: number, edges: number[][]): number[] {
    let map = new Map();
    let seen = new Set();
    let cache = new Map();

    for (let i = 0; i < edges.length; i++) {
      let [x, y] = edges[i];
      if (!map.has(x)) {
        map.set(x, []);
      }
      map.get(x).push([x, y]);

      if (!map.has(y)) {
        map.set(y, []);
      }
      map.get(y).push([x, y]);
    }

    const getKey = ([x, y]: number[]): string => {
      return `${x}-${y}`;
    }

    const dfs = ([x, y]: number[], to: number): number => {

      if (cache.has(`${x}-${y}-${to}`)) {
        return cache.get(`${x}-${y}-${to}`);
      }

      let next = y;
      let count = 0;
      if (y == to && x != to) {
        count++;
        next = x;
      }
      //get neighbors
      let neighbors = map.get(next);
      for (const [x1, y1] of neighbors) {
        if (!seen.has(getKey([x1, y1]))) {
          seen.add(getKey([x1, y1]));
          count += dfs([x1, y1], next);
        }
      }
      cache.set(`${x}-${y}-${to}`, count);
      return count;
    }

    let result = [];
    for (let i = 0; i < n; i++) {
      let count = 0;
      seen = new Set();
      for (const p of map.get(i)) {
        seen.add(getKey(p));
        count += dfs(p, i);
      }
      result.push(count);
    }
    return result;
  };

  minReorder(n: number, connections: number[][]): number {
    let count = 0;
    let seen = new Set();
    let map = new Map();

    for (let i = 0; i < connections.length; i++) {
      let [x, y] = connections[i];
      if (!map.has(x)) {
        map.set(x, []);
      }
      map.get(x).push([x, y]);

      if (!map.has(y)) {
        map.set(y, []);
      }
      map.get(y).push([x, y]);
    }

    const getKey = ([x, y]: number[]): string => {
      return `${x}-${y}`;
    }

    const dfs = ([x, y]: number[], to: number) => {
      let next = x;
      if (y != to && x == to) {
        count++;
        next = y;
      }
      //get neighbors
      let neighbors = map.get(next);
      for (const [x1, y1] of neighbors) {
        if (!seen.has(getKey([x1, y1]))) {
          seen.add(getKey([x1, y1]));
          dfs([x1, y1], next);
        }
      }
    }
    for (const p of map.get(0)) {
      seen.add(getKey(p));
      dfs(p, 0);
    }
    return count;
  };

  numIslands(grid: string[][]): number {
    let m = grid.length; // num of rows
    let n = grid[0].length; //num of cols
    let seen = new Set();

    let directions: number[][] = [[0, 1], [0, -1], [1, 0], [-1, 0]];
    const isValid = (i: number, j: number): boolean => {
      return i < m && i >= 0 && j < n && j >= 0 && grid[i][j] == "1";
    }
    let count = 0;
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let val = grid[i][j];
        let stack = [[i, j]];
        if (val == '1' && !seen.has(`${i}-${j}`)) {
          count++;
          while (stack.length > 0) {
            let [x, y] = stack.pop()!;
            seen.add(`${x}-${y}`);
            for (const [x1, y1] of directions) {
              let nextRow = x + x1;
              let nextCol = y + y1;
              if (isValid(nextRow, nextCol) && !seen.has(`${nextRow}-${nextCol}`)) {
                stack.push([nextRow, nextCol]);
              }
            }
          }
        }
      }
    }
    return count;
  };

  findCircleNum(isConnected: number[][]): number {
    const graph = new Map();
    const n = isConnected.length;
    const seen = new Set();
    const dfs = (node: number) => {
      for (const neighbor of graph.get(node)) {
        if (!seen.has(neighbor)) {
          seen.add(neighbor);
          dfs(neighbor);
        }
      }
    };

    //prepare the data
    for (let i = 0; i < n; i++) {
      if (!graph.has(i)) {
        graph.set(i, []);
      }
      for (let j = i + 1; j < isConnected[i].length; j++) {
        if (!graph.has(j)) {
          graph.set(j, []);
        }
        if (isConnected[i][j]) {
          graph.get(i).push(j);
          graph.get(j).push(i);
        }
      }
    }

    let ans = 0;

    for (let i = 0; i < n; i++) {
      if (!seen.has(i)) {
        ans++;
        seen.add(i);
        dfs(i);
      }
    }

    return ans;
  };

  closestValue(root: TreeNode | null, target: number): number {
    const map = new Map();
    let min = Number.MAX_SAFE_INTEGER;
    const dfs = (node: TreeNode | null) => {
      if (!node) {
        return;
      }
      let val = Math.abs(node.val - target);
      if (val <= min) {
        min = val;
        if (map.has(min)) {
          let oldVal = map.get(min);
          if (oldVal > node.val) {
            oldVal = node.val;
          }
          map.set(min, oldVal);
        } else {
          map.set(min, node.val);
        }
      }

      dfs(node.left);
      dfs(node.right);
    };

    dfs(root);

    return map.get(min);

  };

  insertIntoBST(root: TreeNode | null, val: number): TreeNode | null {
    if (!root) {
      return new TreeNode(val);
    }

    //go right
    if (root.val < val) {
      if (root.right) {
        this.insertIntoBST(root.right, val);
      } else {
        root.right = new TreeNode(val);
      }
    }
    //go left
    if (root.val > val) {
      if (root.left) {
        this.insertIntoBST(root.left, val);
      } else {
        root.left = new TreeNode(val);
      }
    }

    return root;
  }

  isValidBST(root: TreeNode | null): boolean {
    if (!root) {
      return true;
    }

    const bst = (node: TreeNode | null, low: number | null = null, high: number | null = null): boolean => {
      if (!node) {
        return true;
      }


      if ((low && node.val <= low) || (high && node.val >= high)) {
        return false;
      }

      let leftValid = bst(node.left, low, node.val);

      let rightValid = bst(node.right, node.val, high);

      return leftValid && rightValid;

    }

    return bst(root);
  };


  diameterOfBinaryTree(root: TreeNode | null): number {
    let diameter = 0;
    const getMaxPath = (node: TreeNode | null): number => {
      if (!node) {
        return 0;
      }

      let leftMax = getMaxPath(node.left);
      let rightMax = getMaxPath(node.right);
      diameter = Math.max(diameter, leftMax + rightMax);

      return Math.max(leftMax, rightMax) + 1;
    };

    getMaxPath(root);
    return diameter;

  };

  finalPrices(prices: number[]): number[] {
    const stack: number[] = [];
    const result: number[] = [];
    for (let i = prices.length - 1; i >= 0; i--) {
      while (stack.length > 0 && stack[stack.length - 1] > prices[i]) {
        stack.pop();
      }
      let discount = 0;
      if (stack.length > 0) {
        discount = stack[stack.length - 1];
      }
      let finalVal = prices[i] - discount;
      result.push(finalVal);
      stack.push(prices[i]);
    }

    return result.reverse();
  };

  predictPartyVictory(senate: string): string {
    const rQueue = [];
    const dQueue = [];
    for (let i = 0; i < senate.length; i++) {
      if (senate[i] == 'R') {
        rQueue.push(i);
      } else {
        dQueue.push(i);
      }
    }
    let rIndex = 0, dIndex = 0;
    const size = senate.length;
    while (rQueue.length > 0 && dQueue.length > 0) {
      rIndex = rQueue.shift()!;
      dIndex = dQueue.shift()!;
      if (rIndex < dIndex) {
        rQueue.push(rIndex + size);
      } else {
        dQueue.push(dIndex + size);
      }
    }

    if (rQueue.length == 0) {
      return 'Dire';
    } else {
      return 'Radiant';
    }
  };

  asteroidCollision(asteroids: number[]): number[] {
    const stack: number[] = [];
    for (let i = 0; i < asteroids.length; i++) {
      stack.push(asteroids[i]);

      if (stack.length > 1) {
        let top = stack[stack.length - 1];
        let secondTop = stack[stack.length - 2];
        while (stack.length > 1 && secondTop > 0 && top < 0) {
          if (secondTop > Math.abs(top)) {
            stack.pop();
          } else if (secondTop < Math.abs(top)) {
            stack.pop();
            stack.pop();
            stack.push(top);
          } else {
            //same mass, pop both
            stack.pop();
            stack.pop();
          }
          if (stack.length <= 1) {
            break;
          }
          top = stack[stack.length - 1];
          secondTop = stack[stack.length - 2];
        }
      }

    }

    return stack;
  };

  validateStackSequences(pushed: number[], popped: number[]): boolean {
    const temp = [];
    const result = [];
    while (pushed.length > 0 && popped.length > 0) {
      temp.push(pushed.shift());

      while (popped.length > 0 && temp[temp.length - 1] == popped[0]) {
        popped.shift();
        result.push(temp.pop());
      }
    }

    return pushed.length == 0 && popped.length == 0;
  };

  robotWithString(s: string): string {
    const sArray: string[] = s.split('');
    const tStack: string[] = [];
    const minValues: string[] = new Array(s.length).fill('z');//
    minValues[s.length - 1] = sArray[s.length - 1];
    const p: string[] = [];
    let min = 'z';
    for (let i = sArray.length - 1; i >= 0; i--) {
      min = sArray[i] <= min ? sArray[i] : min;
      minValues[i] = min;
    }

    for (let i = 0; i < sArray.length; i++) {
      while (tStack.length && tStack[tStack.length - 1] <= minValues[i]) {
        p.push(tStack.pop()!);
      }
      tStack.push(sArray[i]);
    }

    while (tStack.length > 0) {
      p.push(tStack.pop()!);
    }

    return p.join('');
  };


  nextGreaterElement(nums1: number[], nums2: number[]): number[] {
    const map = new Map();
    const stack = [nums2[0]];
    for (let i = 1; i < nums2.length; i++) {
      while (stack.length > 0 && nums2[i] > stack[stack.length - 1]) {
        let p = stack.pop();
        map.set(p, nums2[i]);
      }
      stack.push(nums2[i]);
    }

    const answer = [];
    for (const num of nums1) {
      answer.push(map.get(num) || -1);
    }
    return answer;
  }

  longestSubarray(nums: number[], limit: number): number {
    //monotonic increasing queue
    const queue1 = [];
    //monotonic decreasing queue
    const queue2 = [];
    let max = 0;
    let left = 0;
    for (let right = 0; right < nums.length; right++) {
      let curr = nums[right];
      while (queue1.length > 0 && curr < nums[queue1[queue1.length - 1]]) {
        queue1.pop();
      }
      queue1.push(right);


      while (queue2.length > 0 && curr > nums[queue2[queue2.length - 1]]) {
        queue2.pop();
      }
      queue2.push(right);

      //when condition violates the rule, move left by incrementing
      while (nums[queue2[0]] - nums[queue1[0]] > limit) {
        while (queue2[0] <= left) {
          queue2.shift();
        }

        while (queue1[0] <= left) {
          queue1.shift();
        }
        left++;
      }
      max = Math.max(max, right - left + 1);
    }

    return max;
  };

  maxSlidingWindow(nums: number[], k: number): number[] {
    let result: number[] = [];
    let queue = [0];
    if (k == 1) {
      return nums;
    }
    for (let i = 1; i < nums.length; i++) { //monotonic decreasing
      if (queue[0] < i - k + 1) {
        queue.shift();
      }

      while (queue.length > 0 && nums[i] > nums[queue[queue.length - 1]]) {
        queue.pop();
      }
      queue.push(i);

      if (i >= k - 1) {
        result.push(nums[queue[0]]);
      }
    }

    return result;
  };

  dailyTemperatures(temperatures: number[]): number[] {
    let stack: number[] = [];
    const n = temperatures.length;
    const answer: number [] = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      let t = temperatures[i];
      while (stack.length > 0 && t > temperatures[stack[stack.length - 1]]) {
        const prevIdx = stack.pop()!;
        answer[prevIdx] = i - prevIdx;
      }
      stack.push(i);
    }

    return answer;
  };

  simplifyPath(path: string): string {
    const stack: string[] = [];
    let arr = path.split('/');
    for (let i = 0; i < arr.length; i++) {
      if (arr[i] == '..') {
        if (stack.length > 0) {
          stack.pop();
        }
      } else if (arr[i] != '.' && arr[i] != '') {
        stack.push(arr[i]);
      }
    }
    return '/' + stack.join('/');
  };

  removeDuplicates(s: string): string {
    const stack: string[] = [];
    for (const n of s) {
      if (stack.length && stack[stack.length - 1] == n) {
        stack.pop();
      } else {
        stack.push(n);
      }
    }

    return stack.join('');
  };

  isValid(s: string): boolean {
    const stack = [];
    let map: any = {
      ')': '(',
      ']': '[',
      '}': '{',
    };
    for (const n of s) {
      let peek = stack.length > 0 ? stack[stack.length - 1] : null;
      let val = map[n];
      if (peek && val && peek == val) {
        stack.pop();
      } else {
        stack.push(n);
      }
    }

    return stack.length == 0;
  };

  oddEvenList(head: ListNode | null): ListNode | null {
    if (!head || !head.next) {
      return head;
    }
    let arr: ListNode[] = [];
    let dummy = new ListNode(-1);
    dummy.next = head;
    let curr = head;
    let prev = dummy;
    let end = null;
    let i = 1;
    while (curr) {
      if (i % 2 == 0) {
        prev.next = curr.next;
        arr.push(curr);
      } else {
        prev = curr;
      }
      curr = curr.next!;
      i++;
    }
    prev.next = arr[0];
    arr[0].next = null;
    for (let i = 1; i < arr.length; i++) {
      arr[i - 1].next = arr[i];
      if (i == arr.length - 1) {
        arr[i].next = null;
      }
    }

    return dummy.next;
  };

  getDecimalValue(head: ListNode | null): number {
    if (!head) {
      return 0;
    }

    const getVal = (node: ListNode | null): { pow: number, sum: number } => {
      if (!node) {
        return {pow: 0, sum: 0};
      }

      let {pow, sum} = getVal(node.next);
      if (node.val == 1) {
        sum = sum + Math.pow(2, pow);
      }

      return {pow: pow + 1, sum};
    }

    return getVal(head).sum;
  };

  reverseEvenLengthGroups(head: ListNode | null): ListNode | null {
    let arr = [];
    let dummy = new ListNode(-1);
    dummy.next = head;
    let curr = head;
    while (curr) {
      arr.push(curr.val);
      curr = curr.next;
    }
    let n = arr.length;
    let i = 0;
    let newHead = head;
    for (let g = 1; g < n; g++) {
      let start = i;
      let end = Math.min(i + g - 1, n - 1);
      let size = end - start + 1;
      for (let idx = end; idx >= start; idx--) {
        if ((size % 2) == 0) {
          //if event group size, reverse
          newHead!.val = arr[idx];
        }
        newHead = newHead!.next;
      }
      i = end + 1;
      if (i >= n) {
        break;
      }
    }

    return dummy.next;
  };

  reverseEvenLengthGroupsArray(head: ListNode | null): ListNode | null {
    let arr = [];
    let dummy = new ListNode(-1);
    dummy.next = head;
    let curr = head;
    while (curr) {
      arr.push(curr.val);
      curr = curr.next;
    }
    let n = arr.length;
    let i = 0;
    let newHead = head;
    for (let g = 1; g < n; g++) {
      let start = i;
      let end = Math.min(i + g - 1, n - 1);
      let size = end - start + 1;
      for (let idx = end; idx >= start; idx--) {
        if ((size % 2) == 0) {
          //if event group size, reverse
          newHead!.val = arr[idx];
        }
        newHead = newHead!.next;
      }
      i = end + 1;
      if (i >= n) {
        break;
      }
    }

    return dummy.next;
  };

  isPalindrome(head: ListNode | null): boolean {
    if (!head) {
      return false;
    }

    let dummy = new ListNode(-1);
    dummy.next = head;
    let slow = head;
    let fast = head;
    let part1 = [];
    while (fast && fast.next) {
      part1.push(slow.val);
      fast = fast.next.next!;
      slow = slow.next!;
      if (fast && !fast.next) {
        slow = slow.next!;
      }
    }
    while (part1.length && slow) {
      let val1 = part1.pop();
      let val2 = slow.val;
      slow = slow.next!;
      if (val1 !== val2) {
        return false;
      }
    }
    return true;
  };

  swapNodes(head: ListNode | null, k: number): ListNode | null {
    if (!head || !head.next) {
      return head;
    }
    let dummy = new ListNode(-1);
    dummy.next = head;

    let first: ListNode | null = head;
    let last: ListNode | null = head;

    let slow: ListNode | null = head, fast: ListNode | null = head;
    for (let i = 1; i < k; i++) {
      fast = fast!.next;
      first = fast;
    }

    while (fast && fast.next) {
      slow = slow!.next;
      last = slow;
      fast = fast.next;
    }

    if (first == last) {
      return dummy.next;
    }

    let temp = first!.val;
    first!.val = last!.val;
    last!.val = temp;

    return dummy.next;
  };

  deleteDuplicates(head: ListNode | null): ListNode | null {
    if (!head || !head.next) {
      return head;
    }
    let dummy = new ListNode(111);
    dummy.next = head;
    let prev = dummy;
    let curr: ListNode | null = head;
    let currVal: number;
    while (curr) {
      let next: ListNode | null = curr.next;
      currVal = curr.val;
      if (next && next.val == currVal) {
        while (curr && curr.val == currVal) {
          prev.next = next;
          curr = next;
          next = next ? next.next : null;
        }
      } else {
        prev = curr;
        curr = next;
      }
    }
    return dummy.next;
  };

  removeNthFromEnd(head: ListNode | null, n: number): ListNode | null {
    if (!head!.next) {
      return null;
    }
    let dummy = new ListNode(0);
    dummy.next = head;
    let nodeToDelete = head;
    let second = head;
    let prevFirst = null;
    while (n > 0 && second) {
      second = second.next;
      n--;
    }

    while (second) {
      second = second.next;
      prevFirst = nodeToDelete;
      nodeToDelete = nodeToDelete!.next;
    }

    if (prevFirst) {
      prevFirst!.next = nodeToDelete!.next;
    } else {
      dummy.next = nodeToDelete!.next;
    }

    return dummy.next;
  };

  deleteMiddle(head: ListNode | null): ListNode | null {
    if (!head!.next) {
      return null;
    }
    let slow = head, fast = head;
    let beforeSlow = slow;
    while (fast && fast.next) {
      fast = fast.next.next;
      beforeSlow = slow;
      slow = slow!.next;
    }
    beforeSlow!.next = slow!.next;
    slow!.next = null;

    return head;
  };

  reverseBetweenRecursion(head: ListNode | null, left: number, right: number): ListNode | null {
    let prev1 = null;
    let start1 = null;
    let end1 = null;
    let node = head;
    let size = right - left + 1;
    let prev = null;
    let curr = null;
    let leftCopy = left;
    let dummy = new ListNode(0);
    dummy.next = head;
    while (node) {
      if (leftCopy > 2) {
        node = node.next;
        leftCopy--;
      } else {
        prev1 = left == 1 ? dummy : node;
        start1 = left == 1 ? node : node.next;

        prev = prev1;
        curr = start1;
        while (size > 0 && curr) {
          let next: ListNode | null = curr.next;
          curr.next = prev;
          prev = curr;
          curr = next;
          size--;
          if (size == 1) {
            end1 = curr;
          }
        }
        prev1.next = prev;
        start1!.next = curr;
        break;
      }
    }
    return dummy.next;
  };

  reverseBetween(head: ListNode | null, left: number, right: number): ListNode | null {
    let prev1 = null;
    let start1 = null;
    let end1 = null;
    let node = head;
    let size = right - left + 1;
    let prev = null;
    let curr = null;
    let leftCopy = left;
    let dummy = new ListNode(0);
    dummy.next = head;
    while (node) {
      if (leftCopy > 2) {
        node = node.next;
        leftCopy--;
      } else {
        prev1 = left == 1 ? dummy : node;
        start1 = left == 1 ? node : node.next;

        prev = prev1;
        curr = start1;
        while (size > 0 && curr) {
          let next: ListNode | null = curr.next;
          curr.next = prev;
          prev = curr;
          curr = next;
          size--;
          if (size == 1) {
            end1 = curr;
          }
        }
        prev1.next = prev;
        start1!.next = curr;
        break;
      }
    }
    return dummy.next;
  };

  pairSumReverse(head: ListNode | null): number {
    if (!head!.next!.next) {
      return head!.val + head!.next!.val;
    }
    let fast = head;
    let slow = head;
    while (fast && fast.next) {
      fast = fast.next.next;
      slow = slow!.next;
    }
    let max = 0;
    let part2 = slow;
    let head2 = null;
    let prev = null;
    while (part2) {
      let next = part2.next;
      part2.next = prev;
      prev = part2;
      part2 = next;
      head2 = prev;
    }
    while (head2 && head) {
      max = Math.max(max, head2!.val + head.val);
      head = head.next;
      head2 = head2.next;
    }
    return max;
  };

  pairSumLoop(head: ListNode | null): number {
    const nums = [];
    while (head) {
      nums.push(head.val);
      head = head.next;
    }
    let i = 0, j = nums.length - 1;
    let max = 0;
    while (i < j) {
      max = Math.max(max, nums[i] + nums[j]);
      i++;
      j--;
    }
    return max;
  };

  swapPairsRecur(head: ListNode | null): ListNode | null {
    // If the list has no node or has only one node left.
    if (head == null || head.next == null) {
      return head;
    }
    // Nodes to be swapped
    let firstNode: ListNode | null = head;
    let secondNode: ListNode | null = head.next;
    // Swapping
    firstNode.next = this.swapPairsRecur(secondNode.next);
    secondNode.next = firstNode;
    // Now the head is the second node
    return secondNode;
  };

  reverseList(head: ListNode | null): ListNode | null {
    if (!head || !head.next) {
      return head;
    }

    let prev = head;
    let curr = head.next;
    let p = this.reverseList(curr)!;
    curr.next = prev;
    prev.next = null;
    return p;
  };

  closeStrings(word1: string, word2: string): boolean {
    if (word1.length != word2.length) {
      return false;
    }
    const map1 = new Map();
    const map2 = new Map();
    let count = 0;
    for (const c of word1) {
      map1.set(c, (map1.get(c) || 0) + 1);
    }
    for (const c of word2) {
      map2.set(c, (map2.get(c) || 0) + 1);
    }
    //compare keys
    for (const k1 of map1.keys()) {
      if (!map2.has(k1)) {
        return false;
      }
    }
    let vals1 = [...map1.values()].sort();
    let vals2 = [...map2.values()].sort();
    for (let i = 0; i < vals1.length; i++) {
      ;
      if (vals1[i] != vals2[i]) {
        return false;
      }
    }
    return true;
  };

  customSortString(order: string, s: string): string {
    const mapS = new Map();
    const mapO = new Map();
    const orderArr = order.split('');
    for (let i = 0; i < s.length; i++) {
      mapS.set(s[i], (mapS.get(s[i]) || 0) + 1);
    }
    for (let i = 0; i < order.length; i++) {
      mapO.set(order[i], i);
    }

    let result = [];
    for (let i = 0; i < orderArr.length; i++) {
      let c = orderArr[i];
      if (mapS.has(c)) {
        //place s chars in order
        let count = mapS.get(c);
        while (count > 0) {
          result.push(c);
          count--;
          if (count == 0) {
            mapS.delete(c);
          }
        }
      }
    }
    for (const [s, c] of mapS) {
      for (let i = 0; i < c; i++) {
        result.push(s);
      }
    }

    return result.join('');

  };

  wordPattern(pattern: string, s: string): boolean {
    const map1 = new Map();
    const map2 = new Map();
    const pArr = pattern.split('');
    const sArr = s.split(' ');

    if (pArr.length !== sArr.length) {
      return false;
    }

    for (let i = 0; i < pArr.length; i++) {
      let p1 = pArr[i];
      let s1 = sArr[i];
      if (!map1.has(p1) && !map2.has(s1)) {
        map1.set(p1, s1);
        map2.set(s1, p1);
      } else if (!(map1.get(p1) == s1 && map2.get(s1) == p1)) {
        return false;
      }
    }

    return true;
  };

  isIsomorphic(s: string, t: string): boolean {
    const map1 = new Map();
    const map2 = new Map();
    const sArr = s.split('');
    const tArr = t.split('');

    for (let i = 0; i < sArr.length; i++) {
      let s1 = sArr[i];
      let t1 = tArr[i];
      if (!map1.has(s1) && !map2.has(t1)) {
        map1.set(s1, t1);
        map2.set(t1, s1);
      } else if (!(map1.get(s1) == t1 && map2.get(t1) == s1)) {
        return false;
      }
    }

    return true;
  };

  checkInclusion(s1: string, s2: string): boolean {
    const map1 = new Map();
    for (let s of s1) {
      map1.set(s, (map1.get(s) || 0) + 1);
    }

    const compare = (map1: Map<string, string>, map2: Map<string, string>): boolean => {
      for (const [k1, v1] of map1) {
        if (!map2.has(k1)) {
          return false;
        }
        let v2 = map2.get(k1)!;
        if (v1 != v2) {
          return false;
        }
      }
      return true;
    }

    let i = 0;
    const map2 = new Map();
    for (let j = 0; j < s2.length; j++) {
      let s = s2[j];
      map2.set(s, (map2.get(s) || 0) + 1);
      if ((j - i + 1) == s1.length) {
        if (compare(map1, map2)) {
          return true;
        }
        //reduce s2[i] count
        let count = map2.get(s2[i]);
        count--;
        if (count <= 0) {
          map2.delete(s2[i]);
        } else {
          map2.set(s2[i], count);
        }
        i++;
      }
    }

    return false;
  };

  maximumUniqueSubarray(nums: number[]): number {
    const map = new Map();
    let maxSum = 0, curr = 0, i = 0;
    for (let j = 0; j < nums.length; j++) {
      curr += nums[j];
      map.set(nums[j], (map.get(nums[j]) || 0) + 1);

      while (map.has(nums[j]) && map.get(nums[j]) > 1 && i <= j) {
        let count = map.get(nums[i]);
        map.set(nums[i], count - 1);
        curr -= nums[i];
        i++;
      }

      maxSum = Math.max(maxSum, curr);
    }

    return maxSum;
  };

  numSubarraysWithSum(nums: number[], goal: number): number {
    const map = new Map();
    let curr = 0, i = 0;
    let count = 0;
    map.set(0, 1);
    for (let j = 0; j < nums.length; j++) {
      curr += nums[j];
      if (map.has(curr - goal)) {
        count += map.get(curr - goal);
      }
      map.set(curr, (map.get(curr) || 0) + 1);
    }

    return count;
  };

  maxSubarrayLength(nums: number[], k: number): number {
    const map = new Map();
    let i = 0, max = 0;
    for (let j = 0; j < nums.length; j++) {
      let num = nums[j];
      let count = (map.get(num) || 0) + 1;
      map.set(num, count);
      if (count <= k) {
        max = Math.max(max, j - i + 1);
      }
      while (count > k) {
        map.set(nums[i], map.get(nums[i]) - 1);
        i++;
        count = map.get(num);
      }
    }

    return max;
  };

  isPathCrossing(path: string): boolean {
    const set = new Set();
    const getKey = (nums: number[]): string => {
      return nums.reduce((acc, curr) => curr + '-' + acc, '');
    }
    let point = [0, 0];
    set.add(getKey(point));
    for (const dir of path) {
      if (dir == 'N') {
        point = [point[0], point[1] + 1];
      }
      if (dir == 'S') {
        point = [point[0], point[1] - 1];
      }
      if (dir == 'E') {
        point = [point[0] + 1, point[1]];
      }
      if (dir == 'W') {
        point = [point[0] - 1, point[1]];
      }
      let key = getKey(point);
      if (set.has(key)) {
        return true;
      }
    }
    return false;
  };

  frequencySort(s: string): string {
    const map = new Map();
    for (const c of s) {
      map.set(c, (map.get(c) || 0) + 1);
    }
    const temp = [...map.entries()];
    temp.sort(([k1, v1], [k2, v2]) => v2 - v1);
    const result: string[] = [];
    for (const [k, v] of temp) {
      for (let i = 0; i < v; i++) {
        result.push(k);
      }
    }
    return result.join('');
  };

  containsDuplicate(nums: number[]): boolean {
    const map = new Map();
    for (const num of nums) {
      map.set(num, (map.get(num) || 0) + 1);
      if (map.get(num) > 1) {
        return false;
      }
    }

    return true;
  };

  lengthOfLongestSubstring(s: string): number {
    const map = new Map();
    let max = 0;
    let left = 0;
    for (let i = 0; i < s.length; i++) {
      let key = s[i];
      if (map.has(key)) {
        max = Math.max(max, i - left);
        while (left < map.get(key) + 1) {
          map.delete(s[left]);
          left++;
        }
      } else {
        max = Math.max(max, i - left + 1);
      }
      map.set(key, i);
    }
    return max;
  };

  equalPairs(grid: number[][]): number {
    const map = new Map();
    let count = 0;
    let n = grid.length;
    const getKey = (nums: number[]): string => {
      return nums.reduce((acc, curr) => curr + '-' + acc, '');
    }
    for (let i = 0; i < n; i++) {
      let row = grid[i];
      let key = getKey(row);
      map.set(key, (map.get(key) || 0) + 1);
    }

    for (let i = 0; i < n; i++) {
      let col = [];
      for (let j = 0; j < n; j++) {
        col.push(grid[j][i]);
      }
      let key = getKey(col);
      if (map.has(key)) {
        count += map.get(key);
      }
    }

    return count;
  };

  maximumSum(nums: number[]): number {
    let max = -1;
    const map = new Map();

    for (let i = 0; i < nums.length; i++) {
      let strArr = (nums[i] + '').split('');
      let sum = strArr.map((c) => parseInt(c)).reduce((acc, curr) => acc + curr, 0);
      let arr = map.get(sum);
      if (!arr) {
        map.set(sum, [nums[i]]);
      } else if (arr.length == 1) {
        if (nums[i] > arr[0]) {
          map.set(sum, [nums[i], arr[0]]);
        } else {
          map.set(sum, [arr[0], nums[i]]);
        }
        max = Math.max(max, arr[0] + nums[i])
      } else {
        arr.pop();
        if (nums[i] > arr[0]) {
          map.set(sum, [nums[i], arr[0]]);
        } else {
          map.set(sum, [arr[0], nums[i]]);
        }
        max = Math.max(max, arr[0] + nums[i])
      }
    }

    return max;
  };

  minimumCardPickup(cards: number[]): number {
    let min = cards.length + 1;
    const map = new Map();
    for (let i = 0; i < cards.length; i++) {
      if (!map.has(cards[i])) {
        map.set(cards[i], [i, 1]);
      } else {
        let [x, y] = map.get(cards[i]);
        map.set(cards[i], [i, y + 1]);
        min = Math.min(i - x + 1, min);
      }
    }
    return min > cards.length ? -1 : min;
  };

  groupAnagrams(strs: string[]): string[][] {
    const result = [];
    const map = new Map();
    for (const s of strs) {
      let temp = s.split('').sort();
      let key = temp.join('');
      map.set(key, (map.get(key) || []).push(s));
    }

    for (const [key, values] of map) {
      result.push(values);
    }

    return result;
  };

  findWinners(matches: number[][]): number[][] {
    const loser = new Map();
    for (const [w, l] of matches) {
      loser.set(l, (loser.get(l) || 0) + 1);
    }
    const part1 = [], part2 = [];

    for (const [k, c] of loser) {
      if (loser.get(k) == 0) {
        part1.push(k);
      }
      if (c == 1) {
        part2.push(k);
      }
    }

    return [part1, part2];

  };

  findMaxLength(nums: number[]): number {
    const map = new Map();
    let n = nums.length;
    let count = 0;
    let max = 0;
    let size = 0;
    map.set(0, -1);
    for (let r = 0; r < n; r++) {
      if (nums[r] == 1) {
        count++;
      } else {
        count--;
      }
      if (map.has(count)) {
        size = r - map.get(count);
        max = Math.max(max, size);
      } else {
        map.set(count, r);// set first occurrence value at index r
      }
    }
    return max;
  };

  numberOfSubarrays(nums: number[], k: number): number {
    const n = nums.length;
    const map = new Map();
    let currOddCount = 0;
    let result = 0;
    map.set(0, 1);

    for (let i = 0; i < n; i++) {
      let odd = nums[i] % 2 == 0 ? 0 : 1;
      currOddCount += odd;
      result += map.get(currOddCount - k) || 0;
      map.set(currOddCount, (map.get(currOddCount) || 0) + 1);
    }
    return result;
  };

  getAverages(nums: number[], k: number): number[] {
    const result = [];
    const prefix = [nums[0]];
    for (let i = 1; i < nums.length; i++) {
      prefix.push(prefix[prefix.length - 1] + nums[i]);
    }

    for (let i = 0; i < nums.length; i++) {
      if (i < k || i > (nums.length - k - 1)) {
        result.push(-1);
      } else {
        let avg = (prefix[i + k] - prefix[i - k] + nums[i - k]) / (1 + 2 * k);
        result.push(Math.trunc(avg));
      }
    }
    return result;
  };

  minOperations(nums: number[], x: number): number {
    const n = nums.length;
    let prefix: number[] = [nums[0]];
    for (let i = 1; i < n; i++) {
      prefix.push(prefix[prefix.length - 1] + nums[i]);
    }

    const reverseSum = prefix[prefix.length - 1] - x;
    if (reverseSum < 0) {
      return -1;
    } else if (reverseSum == 0) {
      return n;
    }
    let l = 0, count = -1;
    let curr = 0;
    for (let r = 0; r < n; r++) {
      curr = prefix[r] - prefix[l] + nums[l];
      while (curr > reverseSum) {
        curr = curr - nums[l];
        l++;
      }
      if (curr == reverseSum) {
        count = Math.max(r - l + 1, count);
      }
    }
    return count == -1 ? -1 : n - count;
  };

  minOperationsDp(nums: number[], x: number): number {
    const n = nums.length;
    const cache = new Map();
    const getKey = (arr: number[], r: number): string => {
      return arr.join('-') + '_' + r;
    }
    const dp = (arr: number[], r: number): number => {
      if (r == 0) {
        return n - arr.length;
      }
      if (r < 0 || arr.length == 0) {
        return n + 1;
      }
      const key = getKey(arr, r);
      if (cache.has(key)) {
        return cache.get(key);
      }
      const removeFromLeft = arr.slice(1);
      const removeFromRight = arr.slice(0, arr.length - 1);

      const val = Math.min(dp(removeFromLeft, r - arr[0]), dp(removeFromRight, r - arr[arr.length - 1]));
      cache.set(key, val);
      return val;
    }
    const result = dp(nums, x);
    return result == n + 1 ? -1 : result;
  };

  subarraysDivByK(nums: number[], k: number): number {
    const map = new Map();
    map.set(0, 1);
    let prefix = 0, count = 0;
    for (let i = 0; i < nums.length; i++) {
      prefix = prefix + nums[i] % k + k; // get positive modulo
      let mod = prefix % k;
      if (map.has(mod)) {
        count += map.get(mod);
      }
      map.set(mod, (map.get(mod) || 0) + 1);
    }

    return count;

  };


  pivotIndex(nums: number[]): number {
    const prefix: number[] = [0];
    for (let i = 0; i < nums.length; i++) {
      prefix.push(prefix[prefix.length - 1] + nums[i]);
    }
    let right = prefix.length - 1;
    for (let i = 1; i <= prefix.length; i++) {
      let diff = prefix[right] - prefix[i];
      if (diff == prefix[i - 1]) {
        return i - 1;
      }
    }
    return -1;
  }

  numSubarrayProductLessThanK(nums: number[], k: number): number {
    let count = 0;
    let prod = 1, i = 0, j = 0;
    while (i <= j && j < nums.length) {
      prod = prod * nums[j];
      while (prod >= k && i <= j) {
        prod = prod / nums[i];
        i++;
      }
      if (prod < k) {
        count += j - i + 1;
        j++;
      }

    }
    return count;
  }

  checkSubarraySum(nums: number[], k: number): boolean {
    const map = new Map();
    let mod = 0;
    map.set(0, -1);
    for (let i = 0; i < nums.length; i++) {
      mod = (nums[i] + mod) % k;
      if (map.has(mod)) {
        let left = map.get(mod);
        if (i - left > 1) {
          return true;
        }
      } else {
        map.set(mod, i);
      }
    }
    return false;
  };

  subarraySum(nums: number[], k: number): number {
    const map = new Map();
    map.set(0, 1);
    let count = 0, curr = 0, j = 0;
    for (let i = 0; i < nums.length; i++) {
      curr += nums[i];
      if (map.has(curr - k)) {
        count += map.get(curr - k);
      }
      map.set(curr, (map.get(curr) || 0) + 1);
    }
    return count;
  };

  maxVowels(s: string, k: number): number {
    const vowels = new Set(['a', 'e', 'i', 'o', 'u']);
    let init = vowels.has(s[0]) ? 1 : 0;
    const prefix: number[] = [init];
    for (let i = 1; i < s.length; i++) {
      const val = vowels.has(s[i]) ? 1 : 0;
      prefix.push(prefix[prefix.length - 1] + val);
    }

    let i = 0, j = Math.min(k - 1, s.length - 1), max = 0;
    let val;
    while (j < s.length) {
      val = prefix[j] - prefix[i] + (vowels.has(s[i]) ? 1 : 0);
      max = Math.max(val, max);
      j++;
      i++;
    }

    return max;
  };

  equalSubstring(s: string, t: string, maxCost: number): number {
    const prefix: number[] = [0];
    for (let i = 0; i < s.length; i++) {
      const diff = Math.abs(t.charCodeAt(i) - s.charCodeAt(i));
      if (i == 0) {
        prefix.push(diff);
      } else {
        const lastVal = prefix[prefix.length - 1];
        prefix.push(lastVal + diff)
      }
    }

    let max = 0, i = 0, j = 0;
    while (i <= j && j <= s.length) {
      let cost = prefix[j] - prefix[i];
      if (maxCost >= cost) {
        max = Math.max(max, j - i);
        j++;
      } else {
        i++;
      }
    }
    return max;
  };

  numSubarrayProductLessThanK1(nums: number[], k: number): number {
    let prod = 1;
    let left = 0;
    let count = 0;
    for (let r = 0; r < nums.length; r++) {
      prod = prod * nums[r];
      while (prod >= k && left <= r) {
        prod = prod / nums[left];
        left++;
      }
      count += r - left + 1;
    }
    return count;
  };

  numSubarray(nums: number[]): number {
    let count = 0;
    for (let r = 0; r < nums.length; r++) {
      for (let l = 0; l < r; l++) {
        if (r > l) {
          count += r - l + 1;
        }
      }
    }
    return count;
  };

  findLength(s: string): number {
    const n = s.length;
    let i = 0;
    let zeros = 0;
    let max = -1;
    for (let j = 0; j < n; j++) {
      if (s[j] == '0') {
        zeros++;
      }

      while (zeros > 1) {
        if (s[i] == '0') {
          zeros--;
        }
        i++;
      }
      max = Math.max(max, j - i + 1);
    }

    return max;
  }


  sortedSquares1(nums: number[]): number[] {
    let minIndex = -1;
    let min = Number.MAX_SAFE_INTEGER;
    const ans = [];
    for (let i = 0; i < nums.length; i++) {
      nums[i] = nums[i] * nums[i];
      if (min > nums[i]) {
        min = nums[i];
        minIndex = i;
      }
    }
    ans.push(nums[minIndex]);
    let i = minIndex - 1;
    let j = minIndex + 1;
    while (i >= 0 && j < nums.length) {
      if (nums[i] > nums[j]) {
        ans.push(nums[j]);
      } else {
        ans.push(nums[i]);
      }
    }
    return ans;

  };

  twoSum(nums: number[], target: number): number[] {
    const n = nums.length;
    nums.sort((a, b) => a - b);
    let i = 0, j = n - 1;

    while (i < j) {
      const temp = nums[i] + nums[j];
      if (temp == target) {
        return [i, j];
      }
      if (temp > target) {
        j--;
      } else {
        i++;
      }
    }
    return [];
  };

  minFallingPathSum(matrix: number[][]): number {
    const m = matrix.length;
    const n = matrix[0].length;
    let memo = new Map();
    const dp = (row: number, col: number): number => {
      if (col >= n || col < 0) {
        return Number.MAX_SAFE_INTEGER;
      }
      if (row == m - 1) {
        return matrix[row][col];
      }

      let key = `${row}-${col}`;
      if (memo.has(key)) {
        return memo.get(key);
      }

      let case1 = dp(row + 1, col - 1);
      let case2 = dp(row + 1, col);
      let case3 = dp(row + 1, col + 1);

      let t = Math.min(case1, case2, case3) + matrix[row][col];
      memo.set(key, t);
      return t;
    }

    let ans = Number.MAX_SAFE_INTEGER;
    for (let i = 0; i < n; i++) {
      ans = Math.min(ans, dp(0, i))
    }
    return ans;
  };

  uniquePathsWithObstacles(obstacleGrid: number[][]): number {
    const m = obstacleGrid.length;
    const n = obstacleGrid[0].length;
    let memo: number[][] = [];
    for (let i = 0; i <= m; i++) {
      memo.push(new Array(n + 1).fill(-1));
    }
    if (obstacleGrid[m - 1][n - 1] == 1 || obstacleGrid[0][0] == 1) {
      return 0;
    }
    const dp = (row: number, col: number): number => {
      if (row == 0 && col == 0) {
        return 1;
      }
      if (row < 0 || col < 0 || obstacleGrid[row][col] == 1) {
        return 0;
      }

      if (memo[row][col] && memo[row][col] != -1) {
        return memo[row][col];
      }

      let leftNext = 0;
      if (col > 0) {
        leftNext = dp(row, col - 1);
      }
      let upNext = 0;
      if (row > 0) {
        upNext = dp(row - 1, col);
      }

      const t = leftNext + upNext;
      memo[row][col] = t;
      return t;
    }

    return dp(m - 1, n - 1);
  };

  uniquePathsWithObstacles1(obstacleGrid: number[][]): number {
    const m = obstacleGrid.length;
    const n = obstacleGrid[0].length;
    let memo: number[][] = [];
    for (let i = 0; i <= m; i++) {
      memo.push(new Array(n + 1).fill(-1));
    }
    if (obstacleGrid[m - 1][n - 1] == 1) {
      return 0;
    }
    const dp = (row: number, col: number): number => {
      if (row == m - 1 && col == n - 1) {
        return 1;
      }
      if (row >= m || col >= n || obstacleGrid[row][col] == 1) {
        return 0;
      }

      if (memo[row][col] && memo[row][col] != -1) {
        return memo[row][col];
      }

      let rightNext = 0;
      if (col < n) {
        rightNext = dp(row, col + 1);
      }
      let downNext = 0;
      if (row < m) {
        downNext = dp(row + 1, col);
      }

      const t = rightNext + downNext;
      memo[row][col] = t;
      return t;
    }

    return dp(0, 0);
  };

  Bfs(grid: number[][]): number {
    const numOfRows = grid.length;
    const numOfCols = grid[0].length;

    const isValid = (x: number, y: number): boolean => {
      return x >= 0 && x < numOfRows && y >= 0 && y < numOfCols && grid[x][y] == 0;
    }
    const seen: boolean[][] = [];
    for (let i = 0; i < numOfRows; i++) {
      seen.push(new Array(numOfCols).fill(false));
    }

    const directions = [[0, 1], [1, 0]];
    const bfs = (r: number, c: number): number => {
      let count = 0;
      let queue: number[][] = [];
      queue.push([r, c]);
      while (queue.length) {
        let nextQueue = [];
        count++;
        for (const [row, col] of queue) {
          if (row == numOfRows - 1 && col == numOfCols - 1) {
            return count;
          }
          for (const [i, j] of directions) {
            let nextRow: number = row + i, nextCol: number = col + j;
            if (isValid(nextRow, nextCol) && !seen[nextRow][nextCol]) {
              seen[nextRow][nextCol] = true;
              nextQueue.push([nextRow, nextCol]);
            }
          }
        }
        queue = nextQueue;
      }

      return -1;
    }
    seen[0][0] = true;
    return bfs(0, 0)
  }

  minPathSum(grid: number[][]): number {
    const numOfRows = grid.length;
    const numOfCols = grid[0].length;
    let memo: number[][] = [];
    for (let i = 0; i <= numOfRows; i++) {
      memo.push(new Array(numOfCols + 1).fill(-1));
    }

    const dp = (i: number, j: number): number => {
      if (i == numOfRows - 1 && j == numOfCols - 1) {
        return grid[i][j];
      }
      if (i >= numOfRows || j >= numOfCols) {
        return Number.MAX_SAFE_INTEGER;
      }

      if (memo[i][j] != -1) {
        return memo[i][j]
      }


      let down = dp(i, j + 1) + grid[i][j];
      let right = dp(i + 1, j) + grid[i][j];

      let val = Math.min(down, right);
      memo[i][j] = val;
      return val;
    }

    return dp(0, 0);
  };

  uniquePaths(m: number, n: number): number {

    let memo: number[][] = [];
    for (let i = 0; i <= m; i++) {
      memo.push(new Array(n).fill(-1));
    }

    const dp = (row: number, col: number): number => {
      if (row == m - 1 && col == n - 1) {
        return 1;
      }

      if (memo[row][col] && memo[row][col] != -1) {
        return memo[row][col];
      }

      let rightNext = 0;
      if (col < n) {
        rightNext = dp(row, col + 1);
      }
      let downNext = 0;
      if (row < m) {
        downNext = dp(row + 1, col);
      }

      const t = rightNext + downNext;
      memo[row][col] = t;
      return t;
    }

    return dp(0, 0);
  };

}
