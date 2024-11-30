import {Component, OnInit} from '@angular/core';
import {Heap} from "../heap";
import {MaxHeap} from "../max-heap";
import {MinHeap} from "../min-heap";

@Component({
  selector: 'app-sample',
  templateUrl: './sample.component.html',
  styleUrls: ['./sample.component.scss']
})
export class SampleComponent implements OnInit {
  title = 'angular15App';

  ngOnInit(): void {
    //const nums = Data.longArray1;
    // console.log(this.minSubArrayLen(7, [2,3,1,2,4,3]));
    //console.log(this.maxVowels('abciiidef', 3));
    // console.log(this.equalSubstring('abcd', 'bcdf', 3));
    // var obj = new NumArray([-2, 0, 3, -5, 2, -1])
    // var param_1 = obj.sumRange(0, 2);
    // var param_2 = obj.sumRange(2, 5);

    // console.log(this.repeatedCharacter('abccbaacz'));
    // console.log(this.checkIfPangram('thequickbrownfoxjum'));
    // console.log(this.checkIfPangram('thequickbrownfoxjumpsoverthelazydogabcdefghijklmnopqrstuvwxyz'));
    // console.log(this.missingNumber([3, 0,1]));
    // console.log(this.missingNumber([9,6,4,2,3,5,7,0,1]));
    // console.log(this.countElements([1,1,3,3,5,5,7,7]));
    // console.log(this.numberOfSubarrays([1,1,2,1,1], 3));
    // console.log(this.numberOfSubarrays([2,4,6], 1));
    // console.log(this.findMaxLength([1, 0, 1, 0, 0, 1, 1, 1]));
    // console.log(this.findMaxLength([0, 1]));
    // console.log(this.findMaxLength([0, 1, 0]));
    // console.log(this.canConstruct('a', 'b'));
    // console.log(this.canConstruct('aa', 'ab'));
    // console.log(this.canConstruct('aa', 'aab'));
    // console.log(this.lengthOfLongestSubstring('nfpdmpi'));
    // console.log(this.lengthOfLongestSubstring('dvdf'));
    // console.log(this.lengthOfLongestSubstring('tmmzuxt'));
    // console.log(this.containsDuplicate([1,2,3,1]));
    // console.log(this.containsDuplicate([1,2,3,4]));
    // let l7 = new ListNode(7, null);
    // let l6 = new ListNode(6, l7);
    // let l5 = new ListNode(5, l6);
    // let l4 = new ListNode(4, l5);
    // let l3 = new ListNode(3, l4);
    // let l2 = new ListNode(2, l3);
    // let l1 = new ListNode(1, l2);
    // // this.reverseBetween(l1, 2, 4);
    // this.reverseBetween1(l1, 3, 6);
    // console.log(this.simplifyPath('/home/'));
    // console.log(this.simplifyPath('/home//foo/'));
    // console.log(this.simplifyPath('/home/user/Documents/../Pictures'));
    // console.log(this.simplifyPath('/../'));
    // console.log(this.simplifyPath('/a/./b/../../c/'));
    // console.log(this.simplifyPath('/.../a/../b///c//../d/./'));
    // console.log(this.makeGood('abBAcC'));
    // console.log(this.makeGood('leEeetcode'));
    // console.log(this.maxSlidingWindow([1,3,-1,-3,5,3,6,7], 3));
    // console.log(this.maxSlidingWindow([7,2,4], 2));
    // console.log(this.maxSlidingWindow([1,-1], 1));
    // console.log(this.maxSlidingWindow([1,3,1,2,0,5], 3));
    // console.log(this.maxSlidingWindow([1], 1));
    // console.log(this.longestSubarray([8,2,4,7], 4));
    // console.log(this.longestSubarray([4,2,2,2,4,4,2,2], 0));
    // console.log(this.nextGreaterElement([4,1,2], [1,3,4,2]));

    // let stockSpanner = new StockSpanner();
    // console.log(stockSpanner.next(100)); // return 1
    // console.log(stockSpanner.next(80));  // return 1
    // console.log(stockSpanner.next(60));  // return 1
    // console.log(stockSpanner.next(70));  // return 2
    // console.log(stockSpanner.next(60));  // return 1
    // console.log(stockSpanner.next(75));  // return 4, because the last 4 prices (including today's price of 75) were less than or equal to today's price.
    // console.log(stockSpanner.next(85));  // return 6

    // console.log(this.robotWithString('zza'));
    // console.log(this.robotWithString('bac'));
    //console.log(this.robotWithString('bfdaceaef'));
    // Number.MAX_SAFE_INTEGER;

    // let root = this.buildTree([2, 2, 2], 0, 3);
    // let root1 = this.buildTree1([5,1,4,null,null,3,6], 0, 7);
    // let root2 = this.buildTree1([5,4,6,null,null,3,7], 0, 7);
    // console.log(this.isValidBST(root));
    // console.log(this.isValidBST(root1));
    // console.log(this.isValidBST(root2));

    //console.log(this.findCircleNum([[1,1,0],[1,1,0],[0,0,1]]));
    // console.log(this.findCircleNum([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1], [0, 0, 0, 0]]));
    // const islands1 = [["1", "1", "1", "1", "0"], ["1", "1", "0", "1", "0"], ["1", "1", "0", "0", "0"], ["0", "0", "0", "0", "0"]];
    // const islands2 = [["1", "1", "0", "0", "0"], ["1", "1", "0", "0", "0"], ["0", "0", "1", "0", "0"], ["0", "0", "0", "1", "1"]];
    // const islands3 = [
    //   ["1", "0", "0", "1", "1", "1", "0", "1", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],
    //   ["1", "0", "0", "1", "1", "0", "0", "1", "0", "0", "0", "1", "0", "1", "0", "1", "0", "0", "1", "0"],
    //   ["0", "0", "0", "1", "1", "1", "1", "0", "1", "0", "1", "1", "0", "0", "0", "0", "1", "0", "1", "0"],
    //   ["0", "0", "0", "1", "1", "0", "0", "1", "0", "0", "0", "1", "1", "1", "0", "0", "1", "0", "0", "1"],
    //   ["0", "0", "0", "0", "0", "0", "0", "1", "1", "1", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],
    //   ["1", "0", "0", "0", "0", "1", "0", "1", "0", "1", "1", "0", "0", "0", "0", "0", "0", "1", "0", "1"],
    //   ["0", "0", "0", "1", "0", "0", "0", "1", "0", "1", "0", "1", "0", "1", "0", "1", "0", "1", "0", "1"],
    //   ["0", "0", "0", "1", "0", "1", "0", "0", "1", "1", "0", "1", "0", "1", "1", "0", "1", "1", "1", "0"],
    //   ["0", "0", "0", "0", "1", "0", "0", "1", "1", "0", "0", "0", "0", "1", "0", "0", "0", "1", "0", "1"],
    //   ["0", "0", "1", "0", "0", "1", "0", "0", "0", "0", "0", "1", "0", "0", "1", "0", "0", "0", "1", "0"],
    //   ["1", "0", "0", "1", "0", "0", "0", "0", "0", "0", "0", "1", "0", "0", "1", "0", "1", "0", "1", "0"],
    //   ["0", "1", "0", "0", "0", "1", "0", "1", "0", "1", "1", "0", "1", "1", "1", "0", "1", "1", "0", "0"],
    //   ["1", "1", "0", "1", "0", "0", "0", "0", "1", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "1"],
    //   ["0", "1", "0", "0", "1", "1", "1", "0", "0", "0", "1", "1", "1", "1", "1", "0", "1", "0", "0", "0"],
    //   ["0", "0", "1", "1", "1", "0", "0", "0", "1", "1", "0", "0", "0", "1", "0", "1", "0", "0", "0", "0"],
    //   ["1", "0", "0", "1", "0", "1", "0", "0", "0", "0", "1", "0", "0", "0", "1", "0", "1", "0", "1", "1"],
    //   ["1", "0", "1", "0", "0", "0", "0", "0", "0", "1", "0", "0", "0", "1", "0", "1", "0", "0", "0", "0"],
    //   ["0", "1", "1", "0", "0", "0", "1", "1", "1", "0", "1", "0", "1", "0", "1", "1", "1", "1", "0", "0"],
    //   ["0", "1", "0", "0", "0", "0", "1", "1", "0", "0", "1", "0", "1", "0", "0", "1", "0", "0", "1", "1"],
    //   ["0", "0", "0", "0", "0", "0", "1", "1", "1", "1", "0", "1", "0", "0", "0", "1", "1", "0", "0", "0"]]
    // console.log(this.numIslands(islands1));
    // console.log(this.numIslands(islands2));
    // console.log('minReorder1', this.minReorder1(6, [[0,1],[1,3],[2,3],[4,0],[4,5]]));
    // console.log('minReorder1', this.minReorder1(6, [[1,0],[1,2],[3,2],[3,4]]));
    // console.log('minReorder1', this.minReorder1(6, [[1,0],[2,0]]));
    // console.log('canVisitAllRooms', this.canVisitAllRooms([[1],[2],[3],[]]));
    // console.log('canVisitAllRooms', this.canVisitAllRooms([[1,3],[3,0,1],[2],[0]]));
    // console.log('findSmallestSetOfVertices', this.findSmallestSetOfVertices(6, [[0,1],[0,2],[2,5],[3,4],[4,2]]));
    // console.log('validPath', this.validPath(6, [[0,1],[0,2],[3,5],[5,4],[4,3]], 0, 5));
    // console.log('validPath', this.validPath(3, [[0,1],[1,2],[2,0]], 0, 2));
    // console.log('countComponents', this.countComponents(6, [[0,1],[0,2],[3,5],[5,4],[4,3]]));
    // console.log('numIslands', this.numIslands(
    //   [[0,0,1,0,0,0,0,1,0,0,0,0,0],
    //         [0,0,0,0,0,0,0,1,1,1,0,0,0],
    //         [0,1,1,0,1,0,0,0,0,0,0,0,0],
    //         [0,1,0,0,1,1,0,0,1,0,1,0,0],
    //         [0,1,0,0,1,1,0,0,1,1,1,0,0],
    //         [0,0,0,0,0,0,0,0,0,0,1,0,0],
    //         [0,0,0,0,0,0,0,1,1,1,0,0,0],
    //         [0,0,0,0,0,0,0,1,1,0,0,0,0]]));

    // console.log('maxAreaOfIsland', this.maxAreaOfIsland(
    //   [[0,0,1,0,0,0,0,1,0,0,0,0,0],
    //         [0,0,0,0,0,0,0,1,1,1,0,0,0],
    //         [0,1,1,0,1,0,0,0,0,0,0,0,0],
    //         [0,1,0,0,1,1,0,0,1,0,1,0,0],
    //         [0,1,0,0,1,1,0,0,1,1,1,0,0],
    //         [0,0,0,0,0,0,0,0,0,0,1,0,0],
    //         [0,0,0,0,0,0,0,1,1,1,0,0,0],
    //         [0,0,0,0,0,0,0,1,1,0,0,0,0]]));
    // console.log('shortestPathBinaryMatrix', this.shortestPathBinaryMatrix([[0]]));
    // console.log('shortestPathBinaryMatrix', this.shortestPathBinaryMatrix([[0, 1], [1, 0]]));
    // console.log('shortestPathBinaryMatrix', this.shortestPathBinaryMatrix([[0, 0, 0], [1, 1, 0], [1, 1, 0]]));
    // console.log('shortestPathBinaryMatrix', this.shortestPathBinaryMatrix([[0, 1, 1, 0, 0, 0], [0, 1, 0, 1, 1, 0], [0, 1, 1, 0, 1, 0], [0, 0, 0, 1, 1, 0], [1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0]]));

    // let root = this.buildTree([3, 5, 1, 6, 2, 0, 8, null, null, 7, 4], 0, 11);
    // let target = this.findNode(root, 5);
    // console.log('distanceK', this.distanceK(root, target, 2));
    // console.log('shortestPath', this.shortestPath([[0,0,0],[1,1,0],[0,0,0],[0,1,1],[0,0,0]], 1));
    // console.log('shortestPath', this.shortestPath([
    //   [0,0,0,0,0,0,0,0,0,0],
    //   [0,1,1,1,1,1,1,1,1,0],
    //   [0,1,0,0,0,0,0,0,0,0],
    //   [0,1,0,1,1,1,1,1,1,1],
    //   [0,1,0,0,0,0,0,0,0,0],
    //   [0,1,1,1,1,1,1,1,1,0],
    //   [0,1,0,0,0,0,0,0,0,0],
    //   [0,1,0,1,1,1,1,1,1,1],
    //   [0,1,0,1,1,1,1,0,0,0],
    //   [0,1,0,0,0,0,0,0,1,0],
    //   [0,1,1,1,1,1,1,0,1,0],
    //   [0,0,0,0,0,0,0,0,1,0]], 1));

    // console.log('nearestExit', this.nearestExit([["+","+",".","+"],[".",".",".","+"],["+","+","+","."]], [1,2]));
    // console.log('nearestExit', this.nearestExit([["+","+","+"],[".",".","."],["+","+","+"]], [1,0]));
    // console.log('nearestExit', this.nearestExit([[".","+"]], [0,0]));
    let board = [
      [-1, -1, -1, -1, -1, -1],
      [-1, -1, -1, -1, -1, -1],
      [-1, -1, -1, -1, -1, -1],
      [-1, 35, -1, -1, 13, -1],
      [-1, -1, -1, -1, -1, -1],
      [-1, 15, -1, -1, -1, -1]];

    let board1 = [
      [-1, 4, -1],
      [6, 2, 6],
      [-1, 3, -1]];
    // console.log('snakesAndLadders', this.snakesAndLadders(board));
    // console.log('snakesAndLadders', this.snakesAndLadders(board1));

    // console.log('openLock', this.openLock(["0201","0101","0102","1212","2002"], "0202"));
    // console.log('calcEquation', this.calcEquation([["a","b"],["b","c"]], [2.0,3.0], [["a","c"], ["b","a"],["a","e"],["a","a"],["x","x"]]));
    // console.log('minMutation', this.minMutation( "AACCGGTT", "AAACGGTA", ["AACCGGTA","AACCGCTA","AAACGGTA"]));
    // console.log('canReach', this.canReach( [4,2,3,0,3,1,2], 5));
    // console.log('maximumDetonation', this.maximumDetonation( [[2,1,3],[6,1,4]]));
    // console.log('maximumDetonation', this.maximumDetonation( [[1,2,3],[2,3,1],[3,4,2],[4,5,3],[5,6,4]]));
    // console.log('ladderLength', this.ladderLength('hit', 'cog', ["hot", "dot", "dog", "lot", "log", "cog"]));
    // console.log('lastStoneWeight', this.lastStoneWeight([2, 7, 4, 1, 8, 1]));
    // console.log('lastStoneWeight', this.lastStoneWeight([2, 2]));
    // console.log('numRescueBoats', this.numRescueBoats([3,2,2,1], 3));
    // console.log('partitionArray', this.partitionArray([3,6,1,2,5], 2));
    // console.log('partitionArray', this.partitionArray([1,2,3], 1));
    // console.log('partitionArray', this.partitionArray([2,2,4,5], 0));
    // console.log('partitionArray', this.partitionArray([3,6,1,2,5], 2));
    // console.log('findMaximizedCapital', this.findMaximizedCapital(2, 0, [1,2,3], [0,1,1]));
    // console.log('findMaximizedCapital', this.findMaximizedCapital(3, 0, [1,2,3], [0,1,2]));
    // console.log('findMaximizedCapital', this.findMaximizedCapital(10, 0, [1,2,3], [0,1,2]));
    // console.log('findMaximizedCapital', this.findMaximizedCapital(2, 0, [1,2,3], [0,9,10]));
    // console.log('findLeastNumOfUniqueInts', this.findLeastNumOfUniqueInts([5,5,4], 1));
    // console.log('findLeastNumOfUniqueInts', this.findLeastNumOfUniqueInts([4,3,1,1,3,3,2], 3));
    // console.log('connectSticks', this.connectSticks([15]));
    // console.log('topKFrequent', this.topKFrequent([1,1,1,2,2,3], 2));
    // let kthLargest = new KthLargest(3, [4,5,8,2]);
    // console.log('kthLargest',  kthLargest.add(3));// return -1
    // console.log('kthLargest',  kthLargest.add(5)); // return 1
    // console.log('kthLargest',  kthLargest.add(10));// return 1
    // console.log('kthLargest',  kthLargest.add(9)); // return 2
    // console.log('kthLargest',  kthLargest.add(4)); // return 3
    // console.log('maximumUnits',  this.maximumUnits([[1,3],[2,2],[3,1]], 4)); // return 8
    // console.log('maximumUnits',  this.maximumUnits([[5,10],[2,5],[4,7],[3,9]], 10)); // return 91
    // console.log('maxNumberOfApples',  this.maxNumberOfApples([100,200,150,1000])); // return 91
    // console.log('minSetSize',  this.minSetSize([3,3,3,3,5,5,5,2,2,7])); // 2
    // console.log('maxIceCream',  this.maxIceCream([1,3,2,4,1], 7)); // return 91
    //console.log('searchMatrix',  this.searchMatrix([[1,3,5,7],[10,11,16,20],[23,30,34,60]], 3)); // return 91
    // console.log('searchMatrix',  this.searchMatrix([[1,3,5,7],[10,11,16,20],[23,30,34,60]], 13)); // return 91
    let spells = [40,11,24,28,40,22,26,38,28,10,31,16,10,37,13,21,9,22,21,18,34,2,40,40,6,16,9,14,14,15,37,15,32,4,27,20,24,12,26,39,32,39,20,19,22,33,2,22,9,18,12,5];
    let potions = [31,40,29,19,27,16,25,8,33,25,36,21,7,27,40,24,18,26,32,25,22,21,38,22,37,34,15,36,21,22,37,14,31,20,36,27,28,32,21,26,33,37,27,39,19,36,20,23,25,39,40];
    // console.log('successfulPairs',  this.successfulPairs(spells, potions, 600));
    // console.log('successfulPairs',  this.successfulPairs([39,34,6,35,18,24,40], [27,37,33,34,14,7,23,12,22,37], 43));
    // console.log('successfulPairs',  this.successfulPairs([2,1,4], [2,3,4,4,4,5,7,8], 8));
    //console.log('answerQueries',  this.answerQueries([2,3,4,5], [1]));
    // console.log('minEatingSpeed',  this.minEatingSpeed([3,6,7,11], 8));
    // console.log('minEatingSpeed',  this.minEatingSpeed([30,11,23,4,20], 5));
    // console.log('minEatingSpeed',  this.minEatingSpeed([30,11,23,4,20], 6));
    // console.log('minimumEffortPath',  this.minimumEffortPath([[1,2,2],[3,8,2],[5,3,5]]));
    // console.log('minimumEffortPath',  this.minimumEffortPath([[1,2,1,1,1],[1,2,1,2,1],[1,2,1,2,1],[1,2,1,2,1],[1,1,1,2,1]]));
    // console.log('minSpeedOnTime',  this.minSpeedOnTime([1,3,2], 6));
    // console.log('minSpeedOnTime',  this.minSpeedOnTime([1,3,2], 2.7));
    // console.log('minSpeedOnTime',  this.minSpeedOnTime([1,3,2], 1.9));
    // console.log('maximizeSweetness',  this.maximizeSweetness([1,2,3,4,5,6,7,8,9], 5));
    // console.log('maximizeSweetness',  this.maximizeSweetness([5,6,7,8,9,1,2,3,4], 8));
    // console.log('permute',  this.permute([1, 2, 3]));

    // console.log('combinationSum',  this.combinationSum([2,3,6,7], 7));
    // console.log('combinationSum',  this.combinationSum([2,3,5], 8));
    // console.log('combinationSum',  this.combinationSum([2], 1));
    // console.log('combinationSum',  this.combinationSum([8,7,4,3], 11));

    // console.log('exist',  this.exist([["A","B","C","C"],["S","F","C","S"],["A","D","E","E"]], 'ABCCED'));
    // console.log('exist',  this.exist([["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], 'SEE'));
    // console.log('exist',  this.exist([["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], 'ABCB'));
    // console.log('generateParenthesis',  this.generateParenthesis(2));
    // console.log('numsSameConsecDiff',  this.numsSameConsecDiff(3, 7));
    // console.log('minCostClimbingStairs',  this.minCostClimbingStairs([10,15,20]));
    // console.log('minCostClimbingStairs',  this.minCostClimbingStairs([1,100,1,1,1,100,1,1,100,1]));
    // console.log('mostPoints',  this.mostPoints([[3, 2], [4, 3], [4, 4], [2, 5]]));
    // console.log('mostPoints',  this.mostPoints([[1,1],[2,2],[3,3],[4,4],[5,5]]));
    // console.log('mostPoints',  this.mostPoints([[43,5]]));
    // console.log('climbStairs',  this.climbStairs(2));
    // console.log('climbStairs',  this.climbStairs(3));
    // console.log('longestCommonSubsequence',  this.longestCommonSubsequence('babcdef', 'abacedf'));

    // console.log('coinChange',  this.coinChange([411,412,413,414,415,416,417,418,419,420,421,422], 9864));
    // console.log('maxProfit',  this.maxProfit(2, [2,4,1]));
    // console.log('maxValueOfCoins',  this.maxValueOfCoins([[1,100,3],[7,8,9]], 2));
    // console.log('maxProfitWithFee',  this.maxProfitWithFee([1,3,2,8,4,9], 2));
    // console.log('maxProfitNoFee',  this.maxProfitNoFee([1,2,3,0,2]));
  }

  maxProfitNoFee(prices: number[]): number {
    const n = prices.length;
    const memo: number[][] = [];
    for(let i = 0; i < n; i++) {
      memo.push([-1, -1]);
    }

    const dp = (i: number, holding: boolean): number => {
      if(i >= n) {
        return 0;
      }
      let holdIdx = holding ? 1 : 0;
      if (memo[i][holdIdx] >= 0) {
        return memo[i][holdIdx];
      }

      let hold = 0, sell = 0, skip = 0;
      if(holding) {
        sell = dp(i+2, !holding) + prices[i] ;
      } else {
        hold = dp(i+1, !holding) - prices[i];
      }
      skip = dp(i + 1, holding);
      const ans = Math.max(hold, sell, skip);
      memo[i][holdIdx] = ans;
      return ans;
    }

    return dp(0, false);
  }

  maxProfitWithFee(prices: number[], fee: number): number {
    const n = prices.length;
    const memo: number[][] = [];
    for(let i = 0; i < n; i++) {
      memo.push([-1, -1]);
    }

    const dp = (i: number, holding: boolean): number => {
      if(i >= n) {
        return 0;
      }
      let holdIdx = holding ? 1 : 0;
      if (memo[i][holdIdx] >= 0) {
        return memo[i][holdIdx];
      }

      let hold = 0, sell = 0, skip = 0;
      if(holding) {
        sell = dp(i+1, !holding) + prices[i] - fee;
      } else {
        hold = dp(i+1, !holding) - prices[i];
      }
      skip = dp(i + 1, holding);
      const ans = Math.max(hold, sell, skip);
      memo[i][holdIdx] = ans;
      return ans;
    }

    return dp(0, false);
  };

  maxValueOfCoins(piles: number[][], k: number): number {
    const n = piles.length;

    const dp = (idx: number, k: number): number => {
      if (idx == n || k == 0) {
        return 0;
      }

      const currentPile = piles[idx];
      let jMax = Math.min(k, currentPile.length);
      let curr = 0;
      let ans = 0;
      //get max from current pile
      for(let j = 0; j < jMax; j++){
        curr += currentPile[j];
        //Max of skip and current pile sum of max elements
        ans = Math.max( dp(idx + 1, k), dp(idx + 1, k - j -1) + curr);
      }
      return ans;
    }
    return dp(0, k);
  };

  maxProfit(k: number, prices: number[]): number {
    const n = prices.length;
    const memo = new Map();

    const getKey = (i: number, ki: number, kj: number): string => {
      return `${i}-${ki}-${kj}`;
    }

    //ki for sell, kj for buy
    const dp = (i: number, ki: number, kj: number): number => {
      //no more sell
      if(i >= n || ki <= 0) {
        return 0;
      }
      const key = getKey(i, ki, kj);
      if (memo.has(key)){
        return memo.get(key);
      }


      let p1 = Number.MIN_SAFE_INTEGER, p2 = Number.MIN_SAFE_INTEGER, p3 = Number.MIN_SAFE_INTEGER
      if (kj < ki) {
        p1 = prices[i] + dp(i+1,ki-1, kj);// sell
      } else if(kj == ki) {
        p2 = dp(i+1, ki, kj-1) - prices[i];// buy
      }
      p3 = dp(i+1,  ki, kj);//no buy no sell

      memo.set(key, Math.max(p1, p2, p3));
      return memo.get(key);
    };

    return dp(0, k, k);
  };

  longestCommonSubsequence(text1: string, text2: string): number {
    const cache = new Map();

    const dp = (i: number, j: number): number => {
      if(i == text1.length || j == text2.length) {
        return 0;
      }

      if(cache.has(`${i}_${j}`)){
        return cache.get(`${i}_${j}`);
      }
      let temp = 0;
      if (text1[i] == text2[j]) {
        temp = dp(i + 1, j + 1) + 1;
      } else {
        temp = Math.max(dp(i, j + 1), dp(i + 1, j));
      }
      cache.set(`${i}_${j}`, temp);
      return temp;
    }

    return dp(0, 0);
  };

  coinChange(coins: number[], amount: number): number {
    if (amount < 1) return 0;
    const cache = new Map();
    const dp = (remaining: number): number => {
      if(remaining == 0) {
        return 0;
      }
      if(remaining < 0) {
        return -1;
      }

      if(cache.has(remaining)) {
        return cache.get(remaining);
      }

      let min = Number.MAX_SAFE_INTEGER;
      for(const coin of coins) {
        let temp = dp(remaining - coin);
        //only increment if the value is valid
        if (temp >= 0 && temp < Number.MAX_SAFE_INTEGER) {
          min = Math.min(min, temp + 1);
        }
      }
      cache.set(remaining, (min == Number.MAX_SAFE_INTEGER) ? -1 : min);
      return cache.get(remaining);
    }

    return dp(amount);
  }
  coinChange1(coins: number[], amount: number): number {
    coins.sort((a, b) => b - a);

    if(amount == 0) {
      return 0;
    }
    if(amount < coins[coins.length - 1]) {
      return -1;
    }

    let answer = Number.MAX_SAFE_INTEGER;

    const backtrack = (idx: number, sum: number, result: number[]) => {
      const n = coins.length;
      if(sum == amount) {
        answer = Math.min(result.length, answer);
        return;
      }
      if(idx >= coins.length) {
        return;
      }
      let remainder = amount - sum;
      for(let i = idx; i < n; i++) {
        const coin = coins[i];
        if(remainder >= coin) {
          result.push(coin);
          backtrack(i, sum + coin, result);
          result.pop();
        }
      }

    }
    backtrack(0, 0, []);
    if(answer == Number.MAX_SAFE_INTEGER) return -1;
    return answer;
  };

  climbStairs(n: number): number {
    const cache: number[] = new Array(n).fill(0);
    const dp = (i: number): number => {
      if(i == n) {
        return 0;
      }
      if(i == n - 1) {
        return 1;
      }
      if(i == n - 2) {
        return 2;
      }

      if(cache[i] > 0) {
        return cache[i];
      }

      let ans = dp(i + 1) + dp(i + 2);
      cache[i] = ans;
      return ans;
    };

    return dp(0);

  };

  mostPoints(questions: number[][]): number {
    const n = questions.length;
    const memo: number[] = new Array(n).fill(-1);
    const dp = (i: number): number => {
      if(i >= n) {
        return 0;
      }

      if(memo[i] > 0) {
        return memo[i];
      }

      let [p, b] = questions[i];
      let nextIndex = i + b + 1;
      memo[i] = Math.max(dp(nextIndex) + p, dp(i + 1));
      return memo[i];
    }

    return dp(0);
  };

  lengthOfLIS1(nums: number[]): number {
    const n = nums.length;
    const memo = new Array(n).fill(1);
    memo[n - 1] = 1;
    for (let i = n - 1; i >= 0; i--) {
      for (let j = n - 1; j >= i; j--) {
        if(nums[i] < nums[j]) {
          memo[i] = Math.max(memo[i], memo[j] + 1);
        }
      }
    }

    return Math.max(...memo);
  }

  lengthOfLIS(nums: number[]): number {
    const n = nums.length;
    const memo = new Array(nums.length).fill(-1);

    const dp = (i: number): number => {
      if(memo[i] >= 0) {
        return memo[i];
      }

      let ans = 1;
      for(let j = n - 1; j >= i; j--){
        if(nums[j] > nums[i]){
          ans = Math.max(dp(j) + 1, ans);
        }
      }
      memo[i] = ans;
      return ans;
    };

    let ans = 0;

    for (let i = n - 1; i >= 0; i--) {
      ans = Math.max(ans, dp(i));
    }

    return ans;

  }

  minCostClimbingStairs(costs: number[]): number {
    const n = costs.length;
    const cache = new Array(costs.length).fill(-1);
    const dp = (i: number): number => {
      if(i > n-1) {
        return 0;
      }

      if(cache[i] > 0) {
        return cache[i];
      }

      let temp = Math.min(dp(i + 1) + costs[i], dp(i + 2) + costs[i]);
      cache[i] = temp;
      return temp;
    }

    return Math.min(dp(0), dp(1));

  };

  rob(nums: number[]): number {
    const cache = new Array(nums.length).fill(-1);
    const robFrom = (i: number, houses: number[]): number => {
      if(i >= houses.length) {
        return 0;
      }
      if(cache[i] >= 0) {
        return cache[i];
      }

      let ans = Math.max(robFrom(i+1, houses), (houses[i] + robFrom(i + 2, houses)))
      cache[i] = ans;
      return ans;
    }

    return robFrom(0, nums);
  };

  combinationSum3(k: number, n: number): number[][] {
    const answer: number[][] = [];

    const isValid = (result: number[], sum: number, i: number): boolean => {
      if(result[result.length - 1] < i) {
        return !result.includes(i) && sum + i <= n && result.length < k;
      }
      return false;
    }

    const backtrack = (num: number, sum: number, result: number[]) => {
      if (sum == n && result.length == k) {
        answer.push([...result]);
        return;
      }
      for (let i = 1; i <= 9; i++) {
        if(isValid(result, sum, i)){
          result.push(i);
          backtrack(i, sum + i, result);
          result.pop();
        }
      }
    }

    for(let i = 1; i <= 9; i++){
      backtrack(i, i,[i]);
    }

    return answer;
  };

  numsSameConsecDiff(n: number, k: number): number[] {
    const answer: number[] = [];

    const backtrack = (num: number, result: number[]) => {
      if (result.length == n) {
        answer.push(parseInt(result.join('')));
        return;
      }
      for (let i = 0; i <= 9; i++) {
        if(Math.abs(i - num) == k){
          result.push(i);
          backtrack(i, result);
          result.pop();
        }
      }
    }

    for(let i = 1; i <= 9; i++){
      backtrack(i, [i]);
    }

    return answer;
  };

  generateParenthesis(n: number): string[] {
    const opens: string[] = [];
    const closes: string[] = [];
    for(let i = 0; i < n; i ++) {
      opens.push('(');
      closes.push(')');
    }
    let answer: string[] = [];
    const isValid = (): boolean => {
      return opens.length <= closes.length;
    }

    const getNext = (str: string): string[] => {
      if(str == '') {
        return ['('];
      }
      if(opens.length > 0 && closes.length > 0) {
        if(opens.length < closes.length){
          return ['(', ')'];
        }
        if(closes.length == opens.length) {
          return ['('];
        }
      }
      if(opens.length == 0 && closes.length > 0){
        return [')'];
      }
      return [];
    }

    const backtrack = (start: string, result: string[]) => {
      if(result.length == 2*n) {
        console.log(result.join(''));
        answer.push(result.join(''));
        return;
      }

      let options = getNext(start);
      for(const opt of options) {
        if(opt == '(' && opens.length > 0) {
          let open = opens.pop()!;
          result.push(open);
          backtrack(open, result);
          result.pop();
          opens.push(open);
        } else if(closes.length > 0){
          let close = closes.pop()!;
          result.push(close);
          backtrack(close, result);
          result.pop();
          closes.push(close);
        }
      }
    }

    backtrack('', []);

    return answer;

  };

  exist(board: string[][], word: string): boolean {
    const entries: number[][] = [];
    const n = board.length;
    const m = board[0].length;
    for(let i = 0; i < n; i ++) {
      for (let j = 0; j < m; j++) {
        if(board[i][j] == word[0]){
          entries.push([i, j]);
        }
      }
    }

    const isValid = (x: number, y: number): boolean => {
      return x >= 0 && x < n && y >= 0 && y < m;
    }
    const options = [[0, 1],[1, 0], [0, -1], [-1, 0]];
    const getDirections = (point: number[]): number[][] => {
      const [a, b] = point;
      const result = [];
      for(const [x, y] of options) {
        let nextRow = a + x;
        let nextCol = b + y;
        if(isValid(nextRow, nextCol)) {
          result.push([nextRow, nextCol]);
        }
      }
      return result;
    }

    const getKey = (x: number, y: number): string => {
      return `${x}-${y}`;
    }

    const backtrack = (seen: Set<any>, index: number, start: number[]): boolean => {
      if(index == word.length) {
        return true;
      }

      const routes = getDirections(start);
      for(const [x, y] of routes) {
        const key = getKey(x, y);
        if(!seen.has(key) && word[index] == board[x][y]) {
          seen.add(key);
          if(backtrack(seen, index + 1, [x, y])){
            return true
          }
          seen.delete(key);
        }
      }

      return false;
    }

    for(let i = 0; i < entries.length; i++){
      let seen = new Set();
      seen.add(getKey(entries[i][0], entries[i][1]));
      if(backtrack(seen, 1, entries[i])) {
        return true;
      }
    }
    return false;
  };

  totalQueens(n: number) {
    let backtrack = (row: number, diagonals: Set<number>, antiDiagonals: Set<number>, cols: Set<number>) => {
      // Base case - N queens have been placed
      if (row == n) {
        return 1;
      }

      let solutions = 0;
      for (let col = 0 ; col < n; col++) {
        let currDiagonal = row - col; // add n to avoid going negative
        let currAntiDiagonal = row + col;

        // If the queen is not placeable
        if (cols.has(col) ||
          diagonals.has(currDiagonal) ||
          antiDiagonals.has(currAntiDiagonal)) {
          continue;
        }

        // "Add" the queen to the board
        cols.add(col);
        diagonals.add(currDiagonal);
        antiDiagonals.add(currAntiDiagonal);

        // Move on to the next row with the updated board state
        solutions += backtrack(row + 1, diagonals, antiDiagonals, cols);

        // "Remove" the queen from the board since we have already
        // explored all valid paths using the above function call
        cols.delete(col);
        if(cols.size == 0) {
          console.log(col, diagonals.size, antiDiagonals.size);
        }
        diagonals.delete(currDiagonal);
        antiDiagonals.delete(currAntiDiagonal);
      }

      return solutions;
    }

    return backtrack(0, new Set(), new Set(), new Set());
  };

  totalNQueens(n: number): number {
    const answer: number[][][] = [];

    const canAttack = (q1: number[], q2: number[]): boolean => {
      if(q1[0] == q2[0] || q1[1] == q2[1]) {
        return true;
      }
      //diagonal
      if((q1[0] - q1[1]) == (q2[0] - q2[1])) {
        return true;
      }
      //an-diagonal
      if((q1[0] + q1[1]) == (q2[0] - q2[1])) {
        return true;
      }

      return false;
    }

    const isValid = (placements: number[][], target: number[]) => {
      for(const curr of placements) {
        if(canAttack(curr, target)) {
          return false;
        }
      }
      return true;
    }

    const isPlaced = (placements: number[][], r: number, c: number) => {
      for(const [x, y] of placements) {
        if(x == r && y == c) {
          return true;
        }
      }
      return false;
    }
    const hasSameValues = (pp1: number[][], pp2: number[][]): boolean => {
      let p1 = [...pp1];
      let p2 = [...pp2];

      p1.sort((a, b) => a[0] - b[0]);
      p2.sort((a, b) => a[0] - b[0]);
      for(let i = 0; i < p1.length; i ++) {
        for(let j = 0; j < p1[0].length; j ++) {
          if(p1[i][j] !== p2[i][j]) {
            return false;
          }
        }
      }
      return true;
    }

    const isDuplicate = (placements: number[][], answer: number[][][]) => {
      for(const p of answer) {
        if(hasSameValues(p, placements)) {
          return true;
        }
      }
      return false;
    }



    const backtrack = (row: number, col: number, placements: number[][]) => {
      if(placements.length == n && !isDuplicate(placements, answer)) {
        answer.push([...placements]);
      }

      for(let i = 0; i < n; i ++) {
        for(let j = 0; j < n; j ++) {
          if(!isPlaced(placements, i, j) && isValid(placements, [i, j])){
            placements.push([i, j]);
            backtrack(i, j, placements);
            placements.pop();
          }
        }
      }
    }

    backtrack(0, 0, []);

    return answer.length;
  };

  combinationSum(candidates: number[], target: number): number[][] {
    const n = candidates.length;
    const answer: number[][] = [];
    const backtrack = (start: number, currentSum: number, path: number[]) => {
      if(currentSum == target) {
        answer.push([...path]);
        return;
      }

      for(let i = start; i < n; i++) {
        let num = candidates[i];
        let temp = currentSum + num;
        if(temp <= target) {
          path.push(num);
          backtrack(i, temp, path);
          path.pop();
        }
      }
    };

    backtrack(0, 0, []);
    return answer;
  };

  letterCombinations(digits: string): string[] {
    const n = digits.length;
    if(n == 0) {
      return [];
    }
    const map = new Map();
    let code = 'a'.charCodeAt(0);
    for(let i = 2; i <= 9; i++) {
      let values = [String.fromCharCode(code), String.fromCharCode(code + 1), String.fromCharCode(code + 2), ]
      map.set(i.toString(), values);
      if(i == 7){
        map.get(i.toString()).push('s');
        code = code + 1;
      }

      if(i == 9){
        map.get(i.toString()).push('z');
      }
      code = code + 3;
    }

    const answer: string[] = [];
    let backtrack = (currIndex: number, result: string[]) => {
      if (currIndex == n) {
        answer.push(result.join(''));
        return;
      }

      const digit = digits[currIndex];
      const letters = map.get(digit);
      for (const letter of letters) {
        result.push(letter);
        backtrack(currIndex + 1, result);
        result.pop();
      }
    }

    backtrack(0, []);
    return answer;
  }

  allPathsSourceTarget(graph: number[][]): number[][] {
    const n = graph.length;
    const answer: number[][] = [];
    let backtrack = (currIndex: number, result: number[]) => {
      if(currIndex == n - 1) {
        answer.push([...result]);
        return;
      }

      const currEdges = graph[currIndex];
      for(const edge of currEdges) {
        if(!result.includes(edge)) {
          result.push(edge);
          backtrack(edge, result);
          result.pop();
        }
      }
    }

    backtrack(0, [0]);
    return answer;
  };

  permute (nums: number[]): number[][] {
    let backtrack = (curr: number[]) => {
      if (curr.length == nums.length) {
        ans.push([...curr]);
        return;
      }

      for (const num of nums) {
        if (!curr.includes(num)) {
          curr.push(num);
          backtrack(curr);
          curr.pop();
        }
      }
    }

    let ans: number[][] = [];
    backtrack([]);
    return ans;
  };
  splitArray(nums: number[], k: number): number {
    const getSplitsRequired = (maxSumAllowed: number): number[] => {
      let sum = 0;
      let splitsRequired = 0;
      let maxSum = 0;
      for(let i = 0; i < nums.length; i++) {
        sum += nums[i];
        if(sum > maxSumAllowed) {
          //splits required
          splitsRequired++;
          maxSum = Math.max(maxSum, sum);
          sum = nums[i];
        }
      }
      return [splitsRequired + 1, maxSum]; //make sure the nums array exhausted
    }


    let left = Math.max(...nums);
    let right = nums.reduce((acc, curr) => {
      acc += curr;
      return acc;
    }, 0);
    let minimumLargestSum = Number.MAX_SAFE_INTEGER;
    while(left <= right) {
      let mid = Math.floor((right + left) / 2);
      let [splits, maxSum] = getSplitsRequired(mid);
      if(splits == k) {
        minimumLargestSum = Math.min(maxSum, minimumLargestSum);
        console.log(`minimumLargestSum=${minimumLargestSum}, maxSUm=${maxSum}, mid=${mid}`);

        right = mid - 1;
      } else if (splits < k) { //if smaller splits needed, then move towards left
        right = mid - 1;
      } else {
        left = mid + 1;
      }
    }
    return left;
  };

  maximizeSweetness(sweetness: number[], k: number): number {
    const verify = (s: number): boolean => {
      let j = 0;
      for(let i = 0; i <= k; i++) {
        let sum = 0;
        while(sum < s) {
          if(j < sweetness.length) {
            sum += sweetness[j];
            j++;
          } else {
            return false;
          }
        }
      }
      return true;

    }

    let left = 1;
    let right = Math.pow(10, 9);
    while(left <= right) {
      let mid = Math.floor((left + right) / 2);
      if(!verify(mid)) {
        right = mid - 1;
      } else {
        left = mid + 1;
      }
    }
    return right;
  };

  smallestDivisor(nums: number[], threshold: number): number {
    const getSum = (divisor: number): number => {
      let sum = 0;
      for(let i = 0; i< nums.length; i++) {
        sum += Math.ceil(nums[i] / divisor);
      }
      return sum;
    }
    let left = 1;
    let right = Math.pow(10, 6);
    while(left <= right) {
      let mid = Math.floor((left + right) / 2);
      let sum = getSum(mid);
      if(sum <= threshold) {
        right = mid - 1;
      } else {
        left = mid + 1;
      }
    }
    return left;
  };

  minSpeedOnTime(dist: number[], hour: number): number {
    let totalDist = dist.reduce((acc, curr) => {
      return acc + curr;
    }, 0);

    if(dist.length > Math.ceil(hour)) {
      return -1;
    }

    const calcTime = (speed: number): number => {
      let t = 0;
      for (let i = 0; i < dist.length; i++) {
        if(i == dist.length - 1) {
          t += dist[i] / speed;
        } else {
          t += Math.ceil(dist[i] / speed);
        }
      }

      return t;
    };

    let left = 1;
    let right =10000000;
    while (left <= right) {
      let mid = Math.floor((left + right) / 2);
      if(calcTime(mid) <= hour) {
        right = mid - 1;
      } else {
        left = mid + 1;
      }
    }
    return left;
  };

  minimumEffortPath(heights: number[][]): number {
    const options: number[][] = [[0, 1], [0, -1], [1, 0], [-1, 0]];
    const numOfRows = heights.length;
    const numOfCols = heights[0].length;
    const isValid = (p: number[]) => {
      const [x, y] = p;
      return x >=0 && y >= 0 && x < numOfRows && y < numOfCols;
    }
    const getNext = (p: number[]): number[][] => {
      let [x, y] = p;
      let result = [];
      for(const [x1, y1] of options) {
        if (isValid([x - x1, y - y1])) {
          result.push([x - x1, y - y1]);
        }
      }
      return result;
    }

    const getKey = (x: number, y: number) => {
      return `${x}-${y}`
    }

    const dfs = (p: number[], min:number, seen: Set<any>): boolean => {
      let [x, y] = p;
      if(!seen.has(getKey(x, y))) {
        seen.add(getKey(x, y));
      }
      if(x == (numOfRows - 1) && y == (numOfCols - 1)) {
        return true;
      }
      for(const n of getNext(p)) {
        let [xn, yn] = n;

        const diff = Math.abs(heights[xn][yn] - heights[x][y]);
        if (diff <= min && !seen.has(getKey(xn, yn))) {
          seen.add(getKey(xn, yn));
          let result = dfs(n, min, seen);
          if(result) {
            return true;
          }
        }
      }
      return false;
    }

    const check = (min:number): boolean => {
      let stack = [[0, 0]];
      const seen = new Set();
      seen.add(getKey(0, 0));
      while (stack.length) {
        const [row, col] = stack.pop()!;
        if(row == (numOfRows - 1) && col == (numOfCols - 1)) {
          return true;
        }

        for(const n of getNext([row, col])) {
          let [x, y] = n;
          const diff = Math.abs(heights[x][y] - heights[row][col]);
          if (diff <= min && !seen.has(getKey(x, y))) {
            seen.add(getKey(x, y));
            stack.push([x, y]);
          }
        }
      }
      return false;
    }

    let left = 0;
    let right = 0;
    for (const arr of heights) {
      right = Math.max(right, Math.max(...arr));
    }

    while (left <= right) {
      let mid = Math.floor((left + right) / 2);
      const seen = new Set();

      // if (check(mid)) {
      if (dfs([0, 0], mid, seen)) {
        right = mid - 1;
      } else {
        left = mid + 1;
      }
    }

    return left;
  };

  minEatingSpeed(piles: number[], h: number): number {
    const calculate = (bananas: number[], speed: number): boolean => {
      let total = 0;
      for (let i = 0; i < bananas.length; i++) {
        let t = Math.ceil(bananas[i] / speed);
        total += t;
      }
      return total <= h;
    }

    let left = 1;
    let right = Math.max(...piles);
    while (left <= right) {
      let mid = Math.ceil((left + right) / 2);
      if (calculate(piles, mid)) {
        right = mid - 1;
      } else {
        left = mid + 1;
      }
    }
    if(calculate(piles, left - 1)) {
      return left - 1;
    }
    return left;

  };

  answerQueries(nums: number[], queries: number[]): number[] {
    nums.sort((a, b) => a - b);
    let prefix: number[] = [nums[0]];
    for(let i = 1; i < nums.length; i++) {
      let sum = prefix[i - 1] + nums[i];
      prefix.push(sum);
    }

    //binary search prefix value
    const getIndex = (target: number, arr: number[]) => {
      let left = 0, right = nums.length - 1;
      while(left < right) {
        let mid = Math.floor((left + right) / 2);
        if(arr[mid] == target) {
          return mid;
        } else if(arr[mid] >= target) {
          right = mid;
        } else {
          left = mid + 1;
        }
      }
      if(left >= arr.length) {
        return left-1;
      }
      if(arr[left] > target) {
        return left-1;
      }
      else {
        return left;
      }
    }

    const result : number[] = [];
    for(let i = 0; i < queries.length; i++) {
      let index = getIndex(queries[i], prefix);
      result.push(index + 1);
    }

    return result;
  };

  searchInsert(nums: number[], target: number): number {
    let left = 0, right = nums.length - 1;
    while(left < right) {
      let mid = Math.floor((left + right) / 2);
      if(nums[mid] >= target) {
        right = mid;
      } else {
        left = mid + 1;
      }
    }
    return left;
  };

  successfulPairs(spells: number[], potions: number[], success: number): number[] {
    const findIndex = (arr: number[], val: number): number => {
      let left = 0, right = arr.length - 1;
      while(left <= right) {
        let mid = Math.floor((left + right) / 2);
        if(arr[mid] >= val) {
          right = mid - 1;
        } else {
          left = mid + 1;
        }
      }
      return left;
    }

    const result: number[] = [];
    potions.sort((a, b) => a - b);
    for(let i = 0; i < spells.length; i++) {
      let s = spells[i];
      let minIndex = findIndex(potions, success / s);
      result.push(potions.length - minIndex);
    }
    return result;
  };

  searchMatrix(matrix: number[][], target: number): boolean {
    let left = 0;
    let right = matrix[0].length - 1;
    let rowSize = matrix[0].length;
    let rowMin = 0;
    let rowMax = matrix.length - 1;
    let colSize = rowMax;

    while(rowMin <= rowMax) {
      let rowMid = Math.floor((rowMin + rowMax) / 2);
      if(matrix[rowMid][0] <= target && matrix[rowMid][rowSize-1] >= target) {
        //search in this row
        let nums = matrix[rowMid];
        while(left <= right) {
          let mid = Math.floor((left + right) / 2);
          if(nums[mid] == target) {
            return true;
          }
          else if (nums[mid] < target) {
            left = mid + 1;
          } else {
            right = mid - 1;
          }
        }
        break;
      }
      else if(matrix[rowMid][0] > target){
        rowMax = rowMid - 1;
      } else {
        rowMin = rowMid + 1;
      }
    }

    return false;
  };

  search(nums: number[], target: number): number {
    let left = 0;
    let right = nums.length - 1;
    while(left <= right) {
      let mid = Math.floor((left + right) / 2);
      if(nums[mid] == target) {
        return mid;
      }
      else if (nums[mid] < target) {
        left = mid + 1;
      } else {
        right = mid - 1;
      }
    }

    return -1;
  };

  maxIceCream(costs: number[], coins: number): number {
    let heap = new Heap((a, b) => costs[a] < costs[b]);
    for(let i = 0; i < costs.length; i++) {
      heap.add(i);
    }

    let sum = 0;
    let count = 0;
    while(sum < coins && heap.size() > 0) {
      let idx = heap.remove()!;
      let cost = costs[idx];
      if((sum + cost) <= coins) {
        sum += cost;
        count++;
      }
    }
    return count;
  };

  minSetSize(arr: number[]): number {
    const map = new Map();
    for(let i = 0; i < arr.length; i++) {
      if (!map.has(arr[i])) {
        map.set(arr[i], 0);
      }
      map.set(arr[i],  map.get(arr[i]) + 1);
    }

    let heap = new MaxHeap();

    for(const k of map.keys()) {
      heap.add(k);
    }

    let size = arr.length;
    let count = 0;
    while(size > (arr.length / 2)) {
      let idx = heap.remove();
      size -= map.get(idx);
      count++;
    }
    return count;
  };

  maxNumberOfApples(weight: number[]): number {
    let heap = new MaxHeap();

    for(let i = 0; i < weight.length; i++) {
      heap.add(weight[i]);
    }

    let sum = 0;
    let remainder = 5000;
    let count = 0;
    while(heap.size() > 0 && remainder > 0) {
      if(heap.peek()! <= remainder) {
        let w = heap.remove()!;
        sum += w;
        remainder -= w;
        count++;
      } else {
        heap.remove();
      }

    }

    return count;
  };

  maximumUnits(boxTypes: number[][], truckSize: number): number {

    let heap = new MaxHeap();

    for(let i = 0; i < boxTypes.length; i++) {
      heap.add(i);
    }

    let sum = 0;
    let totalSize = truckSize;
    while(totalSize > 0 && heap.size() > 0) {
      let index: number = heap.remove()!;
      let [numOfBoxes, numOfUnits] = boxTypes[index];
      if(numOfBoxes <= totalSize) {
        sum += numOfBoxes * numOfUnits;
        totalSize -= numOfBoxes;
      } else {
        sum += totalSize * numOfUnits;
        totalSize = 0;
      }
    }

    return sum;
  };

  maximum69Number (num: number): number {
    let arr: string[] = (num + '').split('');
    let result: string[] = arr;
    for(let i = 0; i < arr.length; i++) {
      if (arr[i] == '6') {
        let temp1: string[] = arr.slice(0, i);
        temp1.push('9');
        let temp2: string[] = arr.slice(i+1);
        result = temp1.concat(temp2);
        break;
      }
    }
    return parseInt(result.join(''));
  };

  topKFrequent(nums: number[], k: number): number[] {
    const map = new Map();
    for(let i = 0; i <nums.length; i++){
      if(!map.has(nums[i])){
        map.set(nums[i], 0);
      }
      let val = map.get(nums[i]);
      map.set(nums[i], val + 1);
    }

    // const values = Array.from(map.values()).sort((a: {key: number, count: number }, b: {key: number, count: number }) => b.count - a.count);
    const heap = new MinHeap();
    for(const key of map.keys()) {
      heap.add(key);
      if(heap.size() > k) {
        heap.remove();
      }
    }

    const result: number[] = [];
    for(let i = 0; i < k; i++){
      result.push(heap.remove()!);
    }
    return result;
  };

  connectSticks(sticks: number[]): number {
    const heap = new MinHeap();
    for(let i = 0; i < sticks.length; i++) {
      heap.add(sticks[i]);
    }

    let cost = 0;
    while(heap.size() > 1) {
      let min = heap.remove()!;
      let min1 = heap.remove()!;
      let newStick = min + min1;
      cost += newStick;
      heap.add(newStick);
    }
    return cost;
  };

  minStoneSum(piles: number[], k: number): number {
    const heap = new MaxHeap();
    let sum = 0;
    for(let i = 0; i < piles.length; i++) {
      heap.add(piles[i]);
      sum += piles[i];
    }
    for(let i = 0; i < k; i++) {
      let val = heap.remove()!;
      let remove = Math.floor(val / 2);
      val = val - remove;
      heap.add(val);
      sum -= remove;
    }

    return sum;
  };

  findLeastNumOfUniqueInts(arr: number[], k: number): number {
    const map = new Map();

    for(let i = 0; i <arr.length; i++){
      if(!map.has(arr[i])){
        map.set(arr[i], {key: arr[i], count: 0});
      }
      let val = map.get(arr[i]);
      map.set(arr[i], {key: arr[i], count: val.count + 1});
    }

    const values = Array.from(map.values()).sort((a: {key: number, count: number }, b: {key: number, count: number }) => b.count - a.count);
    for(let i = 0; i < k; i++) {
      let {key, count} = values[values.length - 1];
      count = count - 1;
      if(count <= 0) {
        values.pop();
      } else {
        values[values.length - 1] = {key, count}
      }
    }
    const set = new Set();
    for(let i = values.length - 1; i >= 0; i--) {
      set.add(values[i]);
    }
    return set.size;

  };

  findMaximizedCapital(k: number, w: number, profits: number[], capital: number[]): number {
    //build map
    const n = profits.length;
    const projects: number[][] = new Array(n);
    for(let i = 0; i < n; i++) {
      projects.push([capital[i], profits[i]]);
    }

    projects.sort((a, b) => a[0] - b[0]);

    const heap = new MaxHeap();
    let idx = 0;
    for(let i = 0; i < k; i++) {
      while(idx < n &&  projects[idx][0] <= w){
        heap.add(projects[idx][1]);//add profits to max heap
        idx++;
      }
      if(heap.size() == 0) {
        return w;
      }

      w += heap.remove()!;
    }
    return w;
  };

  partitionArray(nums: number[], k: number): number {
    nums.sort((a, b) => a - b);
    let curr = 0;
    let count = 1;
    for(let i = 1; i < nums.length; i++) {
      if(nums[i] - nums[curr] > k) {
        count++;
        curr = i;
      }
    }

    return count;
  };

  asteroidsDestroyed(mass: number, asteroids: number[]): boolean {
    asteroids.sort((a, b) => a - b);
    for(let i = 0; i < asteroids.length; i++){
      if(asteroids[i] > mass) {
        return false;
      } else {
        mass += asteroids[i];
      }
    }
    return true;
  };

  numRescueBoats(people: number[], limit: number): number {
    people.sort((a, b) => a - b);
    let steps = 0;
    while(people.length) {
      if(people[people.length - 1] > limit) {
        return -1;
      }
      if(people[people.length - 1] == limit) {
        people.pop();
        steps++;
      } else if(people[people.length - 1] + people[0] <= limit) {
        people.pop();
        people.shift();
        steps++;
      } else {
        people.pop();
        steps++;
      }
    }
    return steps;
  };

  halveArray(nums: number[]): number {
    let sum = 0;
    let heap = new MaxHeap();
    for(const n of nums) {
      heap.add(n);
      sum += n;
    }
    let total = sum;
    let steps = 0;
    while(heap.size() > 0 && total > sum / 2) {
      let temp = heap.remove()! / 2;
      heap.add(temp);
      total -= temp;
      steps++;
    }

    return steps;
  };

  lastStoneWeight(stones: number[]): number {
    const heap = new MaxHeap();
    for (const s of stones) {
      heap.add(s);
    }
    //stones.sort((a: number, b: number) => a - b);
    while (heap.size() > 1) {
      let first: number = heap.remove()!;
      if(heap.size() == 0) {
        return first;
      }
      let second = heap.peek()!;
      let next = Math.abs(first - second);
      heap.remove();
      if (next > 0) {
        heap.add(next);
      } else {
      }
    }

    return heap.peek() || 0;
  };

  testMinHeap() {

    // Creating the Heap
    var heap = new MinHeap();

    // Adding The Elements
    heap.add(10);
    heap.printHeap();
    heap.add(15);
    heap.printHeap();
    heap.add(30);
    heap.printHeap();
    heap.add(40);
    heap.printHeap();
    heap.add(50);
    heap.printHeap();
    heap.add(100);
    heap.printHeap();
    heap.add(40);
    // Printing the Heap
    heap.printHeap();

    // Peeking And Removing Top Element
    console.log(heap.peek());
    console.log(heap.remove());

    // Printing the Heap
    // After Deletion.
    heap.printHeap();
  }

  testHeap() {

    // Creating the Heap
    var heap = new Heap();

    // Adding The Elements
    heap.add(10);
    heap.printHeap();
    heap.add(15);
    heap.printHeap();
    heap.add(30);
    heap.printHeap();
    heap.add(40);
    heap.printHeap();
    heap.add(50);
    heap.printHeap();
    heap.add(100);
    heap.printHeap();
    heap.add(9);
    // Printing the Heap
    heap.printHeap();

    // Peeking And Removing Top Element
    console.log(heap.peek());
    console.log(heap.remove());

    // Printing the Heap
    // After Deletion.
    heap.printHeap();
  }

  ladderLength(beginWord: string, endWord: string, wordList: string[]): number {
    if (beginWord.length != endWord.length) {
      return -1;
    }
    const compare = (s1: string, s2: string): number => {
      let count: number = 0;
      for (let i = 0; i < s1.length; i++) {
        if (s1.charAt(i) != s2.charAt(i)) {
          count++;
        }
      }
      return count;
    }
    const getNext = (s: string): string[] => {
      const sets = [];
      for (let i = 0; i < wordList.length; i++) {
        let t = wordList[i];
        if (compare(s, t) == 1) {
          sets.push(t);
        }
      }
      return sets;
    }

    let count = 0;
    let seen = new Set();
    let queue: string[] = [beginWord];
    while (queue.length) {
      let nextQueue: string[] = [];
      count++;
      for (const curr of queue) {
        if (curr == endWord) {
          return count;
        }

        for (const next of getNext(curr)) {
          if (!seen.has(next)) {
            seen.add(next);
            nextQueue.push(next);
          }
        }
      }
      queue = nextQueue;
    }

    return 0;
  };

  maximumDetonation(bombs: number[][]): number {
    const map = new Map();

    //build map
    for (let i = 0; i < bombs.length; i++) {
      let [x, y, r] = bombs[i];
      if (!map.has(i)) {
        map.set(i, []);
      }
      for (let j = 0; j < bombs.length; j++) {
        if (i == j) {
          continue;
        }
        let [x1, y1, r1] = bombs[j];
        let temp = Math.sqrt((x1 - x) * (x1 - x) + (y1 - y) * (y1 - y));
        if (r >= temp) {
          map.get(i).push(j);
        }
      }
    }

    const countMax = (idx: number): number => {
      let queue: number[] = [idx];
      let seen: boolean[] = new Array(bombs.length).fill(false);
      seen[idx] = true;
      let count = 0;
      while (queue.length) {
        let nextQueue: number[] = [];
        for (const idx of queue) {
          count++;
          for (const next of map.get(idx)) {
            if (!seen[next]) {
              seen[next] = true;
              nextQueue.push(next);
            }
          }
        }
        queue = nextQueue;
      }
      return count;
    }

    let result = 0;

    for (let i = 0; i < bombs.length; i++) {
      result = Math.max(countMax(i), result)
    }

    return result;
  };

  canReach(arr: number[], start: number): boolean {

    const getNext = (i: number): number[] => {
      const sets = [];
      if ((i + arr[i]) < arr.length) {
        sets.push(i + arr[i]);
      }
      if (i - arr[i] >= 0) {
        sets.push(i - arr[i]);
      }
      return sets;
    }

    let seen: boolean[] = new Array(arr.length).fill(false);
    seen[start] = true;

    let steps = 0;
    let queue: number[] = [start];
    while (queue.length) {
      let nextQueue: number[] = [];
      for (const curr of queue) {
        if (arr[curr] == 0) {
          return true;
        }
        for (const next of getNext(curr)) {
          if (!seen[next]) {
            seen[next] = true;
            nextQueue.push(next);
          }
        }
      }

      steps++;
      queue = nextQueue;
    }

    return false;
  };

  minMutation(startGene: string, endGene: string, bank: string[]): number {
    if (startGene.length != endGene.length) {
      return -1;
    }
    const choices = ['A', 'C', 'G', 'T'];
    const getNext = (s: string): string[] => {
      const sets = [];
      for (let i = 0; i < startGene.length; i++) {
        let curr = s.charAt(i);
        for (let j = 0; j < choices.length; j++) {
          let nextChar = choices[j];
          if (curr == nextChar) {
            continue;
          }
          let nextStr = s.slice(0, i) + nextChar + s.slice(i + 1);
          sets.push(nextStr);
        }
      }
      return sets;
    }

    let validGenes = new Set();
    for (let s of bank) {
      validGenes.add(s);
    }

    let steps = 0;
    let seen = new Set();
    seen.add(startGene);
    let queue: string[] = [startGene];
    while (queue.length) {
      let nextQueue: string[] = [];
      for (const curr of queue) {
        if (curr == endGene) {
          return steps;
        }

        for (const next of getNext(curr)) {
          if (!seen.has(next) && validGenes.has(next)) {
            seen.add(next);
            nextQueue.push(next);
          }
        }
      }

      steps++;
      queue = nextQueue;
    }

    return -1;
  };

  calcEquation(equations: string[][], values: number[], queries: string[][]): number[] {
    const graph = new Map();
    for (let i = 0; i < equations.length; i++) {
      let [a, b] = equations[i];
      if (!graph.has(a)) {
        graph.set(a, new Map());
      }
      graph.get(a).set(b, values[i]);

      if (!graph.has(b)) {
        graph.set(b, new Map());
      }
      graph.get(b).set(a, 1 / values[i]);
    }

    const calc = ([x, y]: string[]): number => {
      const seen = new Set();

      if (x == y) {
        if (graph.has(x)) {
          return 1;
        } else {
          return -1;
        }
      }

      let queue = [{key: x, result: 1}];
      while (queue.length) {
        let nextQueue: { key: string, result: number }[] = [];
        for (let {key, result} of queue) {
          if (seen.has(key)) {
            continue;
          }
          seen.add(key);
          if (key == y) {
            return result;
          }
          let nextMap = graph.get(key);
          if (nextMap) {
            for (const newKey of nextMap.keys()) {
              if (!seen.has(newKey)) {
                let newAns = nextMap.get(newKey) * result;
                nextQueue.push({key: newKey, result: newAns});
              }
            }
          }
        }
        queue = nextQueue;
      }
      return -1;
    }

    let ans: number[] = [];
    for (const q of queries) {
      ans.push(calc(q));
    }
    return ans;
  };

  openLock(deadends: string[], target: string): number {
    if (target == '0000') {
      return -1;
    }

    const seen = new Map();
    for (const s of deadends) {
      if (s == '0000') {
        return -1;
      }
      seen.set(s, true);
    }
    const nextSets = (s: string): string[] => {
      const sets = [];
      for (let i = 0; i < 4; i++) {
        let n = parseInt(s.charAt(i));
        let nextNum1 = (n + 1) % 10;
        let nextNum2 = (n - 1) < 0 ? 9 : n - 1;
        let nextS1 = s.slice(0, i) + nextNum1 + s.slice(i + 1);
        let nextS2 = s.slice(0, i) + nextNum2 + s.slice(i + 1);
        sets.push(nextS1);
        sets.push(nextS2);
      }
      return sets;
    }
    let queue = ['0000'];
    let steps = 0;
    while (queue.length) {
      let nextQueue: string[] = [];
      for (let i = 0; i < queue.length; i++) {
        let s = queue[i];
        if (s == target) {
          return steps;
        }
        for (const item of nextSets(s)) {
          if (!seen.has(item)) {
            seen.set(item, true);
            nextQueue.push(item);
          }
        }
      }
      steps++;
      queue = nextQueue;
    }

    return -1;
  };

  snakesAndLadders(board: number[][]): number {
    const map = new Map();
    const n = board.length;

    let num = 1;
    let v = 0;
    for (let i = n - 1; i >= 0; i--) {
      if (v % 2 == 0) {
        for (let j = 0; j < n; j++) {
          map.set(num, [i, j]);
          num++;
        }
      } else {
        for (let j = n - 1; j >= 0; j--) {
          map.set(num, [i, j]);
          num++;
        }
      }
      v++;
    }

    //index is the number of each square 1 to NxN
    const getNext = (curr: number): number[] => {
      const nextArray: number[] = [];

      for (let i = 1; i <= 6; i++) {
        let next = curr + i;
        if (next > n * n) {
          break;
        }
        let [x, y] = map.get(next);
        let num = board[x][y];
        if (num != -1) {//snake
          nextArray.push(num);
        } else {
          nextArray.push(next);
        }
      }
      return nextArray;
    }

    const seen: boolean[] = [];
    for (let i = 1; i <= n * n; i++) {
      seen.push(false);
    }

    seen[0] = true;

    let steps = 0;
    let queue: number[] = [1];
    while (queue.length) {
      let nextQueue = [];
      for (let item of queue) {
        if (item == n * n) {
          return steps;
        }
        for (const nextNum of getNext(item)) {
          if (!seen[nextNum]) {
            seen[nextNum] = true;
            nextQueue.push(nextNum);
          }
        }
      }
      steps++;
      queue = nextQueue;
    }

    return -1;
  };


  nearestExit(maze: string[][], entrance: number[]): number {
    const directions = [[0, 1], [0, -1], [1, 0], [-1, 0]];
    const numOfRows = maze.length;
    const numOfCols = maze[0].length;
    const valid = (i: number, j: number): boolean => {
      return i >= 0 && i < numOfRows && j >= 0 && j < numOfCols && maze[i][j] == ".";
    }

    const isValidExit = (i: number, j: number): boolean => {
      if (i == entrance[0] && j == entrance[1]) {
        return false;
      }
      return valid(i, j) && ((i == 0 || i == numOfRows - 1) || (j == 0 || j == numOfCols - 1));
    }

    const seen: boolean[][] = [];
    for (let i = 0; i < numOfRows; i++) {
      seen.push(new Array(numOfCols).fill(false));
    }

    seen[entrance[0]][entrance[1]] = true;

    let steps = 0;
    let queue: number[][] = [[entrance[0], entrance[1]]];
    while (queue.length) {
      let nextQueue = [];
      for (let [row, col] of queue) {
        if (isValidExit(row, col)) {
          return steps;
        }
        for (const [x, y] of directions) {
          let nextRow = row + x, nextCol = col + y;
          if (valid(nextRow, nextCol) && !seen[nextRow][nextCol]) {
            seen[nextRow][nextCol] = true;
            nextQueue.push([nextRow, nextCol]);
          }
        }
      }
      steps++;
      queue = nextQueue;
    }

    return -1;
  };

  shortestPath(grid: number[][], k: number): number {
    const n: number = grid.length;
    const m: number = grid[0].length;
    const directions = [[0, 1], [0, -1], [1, 0], [-1, 0]];

    let walls = [];
    const seen: number[][] = [];
    for (let i = 0; i < n; i++) {
      seen.push(new Array(m).fill(-1));
    }

    let queue: number[][] = [[0, 0, k]];
    const isValid = (x: number, y: number): boolean => {
      return x >= 0 && x < n && y >= 0 && y < m;
    }
    let level = 0;

    while (queue.length) {
      let nextQueue: number[][] = [];
      for (let [row, col, remain] of queue) {
        if (row == n - 1 && col == m - 1) {
          return level;
        }
        if (grid[row][col] == 1) {
          if (remain == 0) { // this path is blocked
            continue;
          } else {
            remain--;
          }
        }
        // if the square has already been visited, but with more removals,
        // then the current path is backwards, remain large to small is correct direction
        if (seen[row][col] >= remain) {
          continue;
        }

        seen[row][col] = remain;

        for (const [x, y] of directions) {
          let nextRow = row + x, nextCol = col + y;
          if (isValid(nextRow, nextCol)) {
            nextQueue.push([nextRow, nextCol, remain]);
          }
        }
      }
      level++;
      queue = nextQueue;
    }

    return -1;
  };

  updateMatrix(mat: number[][]): number[][] {
    const numOfRows = mat.length;
    const numOfCols = mat[0].length;
    const directions = [[0, 1], [0, -1], [1, 0], [-1, 0]];
    let queue: number[][] = [];

    for (let i = 0; i < numOfRows; i++) {
      for (let j = 0; j < numOfCols; j++) {
        if (mat[i][j] == 0) {
          queue.push([i, j]);
        } else {
          mat[i][j] = -1;
        }
      }
    }

    const ans: number[][] = [];
    for (let i = 0; i < numOfRows; i++) {
      ans.push(new Array(numOfCols).fill(-1));
    }

    const isValid = (x: number, y: number): boolean => {
      return x >= 0 && x < numOfRows && y >= 0 && y < numOfCols && mat[x][y] == -1;
    }
    let level = 0;
    while (queue.length) {
      let nextQueue: number[][] = [];
      level++;
      for (const [row, col] of queue) {
        for (const [x, y] of directions) {
          let nextRow = row + x, nextCol = col + y;
          if (isValid(nextRow, nextCol) && mat[nextRow][nextCol] == -1) {
            mat[nextRow][nextCol] = level;
            nextQueue.push([nextRow, nextCol]);
          }
        }
      }
      queue = nextQueue;
    }
    return mat;
  };

  distanceK(root: TreeNode | null, target: TreeNode | null, k: number): number[] {
    const graph = new Map();
    const build = (node: TreeNode | null) => {
      if (!node) {
        return;
      }
      if (node.left) {
        if (!graph.has(node.val)) {
          graph.set(node.val, []);
        }
        graph.get(node.val).push(node.left!.val);
        if (!graph.has(node.left!.val)) {
          graph.set(node.left!.val, []);
        }
        graph.get(node.left!.val).push(node.val);
        build(node.left!);
      }
      if (node.right) {
        if (!graph.has(node.val)) {
          graph.set(node.val, []);
        }
        graph.get(node.val).push(node.right!.val);
        if (!graph.has(node.right!.val)) {
          graph.set(node.right!.val, []);
        }
        graph.get(node.right!.val).push(node.val);
        build(node.right!);
      }
    };

    build(root);

    const seen: boolean[] = new Array(graph.size).fill(false);
    let queue: number[] = [target!.val];
    let level: number = 0;
    let ans: number[] = [];
    while (queue.length) {
      if (level > k) {
        break;
      }
      let nextQueue: number[] = [];
      for (const n of queue) {
        if (!seen[n]) {
          seen[n] = true;
          if (level == k) {
            ans.push(n);
          }
          for (const m of graph.get(n)) {
            nextQueue.push(m);
          }
        }
      }
      level++;
      queue = nextQueue;
    }

    return ans;
  };

  shortestPathBinaryMatrix(grid: number[][]): number {
    const numOfRows = grid.length;
    const numOfCols = grid[0].length;

    if (grid[0][0] != 0 || grid[numOfRows - 1][numOfCols - 1] != 0) {
      return -1;
    }
    const isValid = (x: number, y: number): boolean => {
      return x >= 0 && x < numOfRows && y >= 0 && y < numOfCols && grid[x][y] == 0;
    }
    const seen: boolean[][] = [];
    for (let i = 0; i < numOfRows; i++) {
      seen.push(new Array(numOfCols).fill(false));
    }

    const directions = [[0, 1], [0, -1], [1, 1], [1, -1], [1, 0], [-1, 1], [-1, -1], [-1, 0]];
    const bfs = (r: number, c: number): number => {
      let count = 0;
      let queue: number[][] = [[]];
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
  };

  maxAreaOfIsland1(grid: number[][]): number {
    const matrix = [[0, -1], [0, 1], [1, 0], [-1, 0]];

    let max = 0;
    let numOfRows = grid.length;
    let numOfCols = grid[0].length;

    const isValid = (r: number, c: number): boolean => {
      return r >= 0 && r < numOfRows && c >= 0 && c < numOfCols && grid[r][c] == 1;
    }

    const dfs = (row: number, col: number): number => {
      let area = 0;

      if (isValid(row, col)) {
        area++;
        grid[row][col] = 0
      }

      for (const [x, y] of matrix) {
        let nextRow = row + x;
        let nextCol = col + y;
        if (isValid(nextRow, nextCol)) {
          area += dfs(nextRow, nextCol);
        }
      }
      return area;
    }

    for (let i = 0; i < numOfRows; i++) {
      for (let j = 0; j < numOfCols; j++) {
        if (grid[i][j] == 1) {
          let a = dfs(i, j);
          max = Math.max(a, max);
        }
      }
    }
    return max;
  };


  numIslands(grid: number[][]): number {
    let numOfRows = grid.length;
    let numOfCols = grid[0].length;

    let ans = 0;
    let seen: boolean[][] = [];
    for (let i = 0; i < numOfRows; i++) {
      seen.push(new Array(numOfCols).fill(false));
    }
    const isValid = (row: number, col: number): boolean => {
      return row >= 0 && col >= 0 && row < numOfRows && col < numOfCols && grid[row][col] == 1;
    }

    let directions = [[0, 1], [1, 0], [0, -1], [-1, 0]];

    const dfs2 = (row: number, col: number) => {
      const stack: number[][] = [];
      stack.push([row, col]);
      while (stack.length) {
        let [r, c] = stack.pop()!;
        for (const [i, j] of directions) {
          let nextRow = r + i;
          let nextCol = c + j;
          if (isValid(nextRow, nextCol) && !seen[nextRow][nextCol]) {
            seen[nextRow][nextCol] = true;
            stack.push([nextRow, nextCol]);
          }
        }
      }

    }

    const dfs = (row: number, col: number) => {
      for (const [i, j] of directions) {
        let nextRow = row + i;
        let nextCol = col + j;
        if (isValid(nextRow, nextCol) && !seen[nextRow][nextCol]) {
          seen[nextRow][nextCol] = true;
          dfs(nextRow, nextCol);
        }
      }
    }

    const dfs1 = (row: number, col: number) => {
      //check neighbors
      if (isValid(row - 1, col) && !seen[row - 1][col]) {
        seen[row - 1][col] = true;
        dfs(row - 1, col);
      }
      if (isValid(row + 1, col) && !seen[row + 1][col]) {
        seen[row + 1][col] = true;
        dfs(row + 1, col);
      }
      if (isValid(row, col - 1) && !seen[row][col - 1]) {
        seen[row][col - 1] = true;
        dfs(row, col - 1);
      }
      if (isValid(row, col + 1) && !seen[row][col + 1]) {
        seen[row][col + 1] = true;
        dfs(row, col + 1);
      }

      // for (const [dx, dy] of directions) {
      //   let nextRow = row + dy, nextCol = col + dx;
      //   if (isValid(nextRow, nextCol) && !seen[nextRow][nextCol]) {
      //     seen[nextRow][nextCol] = true;
      //     dfs(nextRow, nextCol);
      //   }
      // }
    }

    for (let i = 0; i < numOfRows; i++) {
      for (let j = 0; j < numOfCols; j++) {
        if (!seen[i][j] && grid[i][j] == 1) {
          ans++;
          seen[i][j] = true;
          dfs(i, j);
        }
      }
    }

    return ans;

  };

  maxAreaOfIsland(grid: number[][]): number {
    const matrix = [[0, -1], [0, 1], [1, 0], [-1, 0]];

    let max = 0;
    let numOfRows = grid.length;
    let numOfCols = grid[0].length;

    const seen: boolean[][] = [];
    for (let i = 0; i < numOfRows; i++) {
      seen.push(new Array(numOfCols).fill(false));
    }
    const isValid = (r: number, c: number): boolean => {
      return r >= 0 && r < numOfRows && c >= 0 && c < numOfCols && grid[r][c] == 1;
    }

    const dfs = (row: number, col: number): number => {
      let area = 0;

      if (isValid(row, col) && !seen[row][col]) {
        area++;
        seen[row][col] = true;
      }

      for (const [x, y] of matrix) {
        let nextRow = row + x;
        let nextCol = col + y;
        if (isValid(nextRow, nextCol) && !seen[nextRow][nextCol]) {
          area += dfs(nextRow, nextCol);
        }
      }
      return area;
    }

    for (let i = 0; i < numOfRows; i++) {
      for (let j = 0; j < numOfCols; j++) {
        if (!seen[i][j] && grid[i][j] == 1) {
          let a = dfs(i, j);
          max = Math.max(a, max);
        }
      }
    }
    return max;
  };

  countComponents(n: number, edges: number[][]): number {
    const map = new Map();

    for (let [from, to] of edges) {
      if (!map.has(to)) {
        map.set(to, []);
      }
      if (!map.has(from)) {
        map.set(from, []);
      }
      map.get(to).push(from);
      map.get(from).push(to);
    }
    for (let i = 0; i < n; i++) {
      if (!map.has(i)) {
        map.set(i, []);
      }
    }


    const seen: boolean[] = new Array(n).fill(false);

    let count = 0;
    const walk = (v: number) => {
      if (!seen[v]) {
        seen[v] = true;
        for (const t of map.get(v)) {
          walk(t);
        }
        map.delete(v);
      }
    }
    let keys = Array.from(map.keys());
    while (keys.length > 0) {
      let k = keys.pop()
      walk(k);
      keys = Array.from(map.keys());
      count++;
    }

    return count;
  };

  validPath(n: number, edges: number[][], source: number, destination: number): boolean {
    if (!edges.length && source == destination) {
      return true;
    }

    const map = new Map();
    for (let [from, to] of edges) {
      if (!map.has(to)) {
        map.set(to, []);
      }
      if (!map.has(from)) {
        map.set(from, []);
      }
      map.get(to).push(from);
      map.get(from).push(to);
    }
    const seen: boolean[] = new Array(n).fill(false);

    const walk = (v: number) => {
      if (!seen[v]) {
        seen[v] = true;
        for (const t of map.get(v)) {
          walk(t);
        }
      }
    }
    walk(source);
    return seen[destination];
  };

  findSmallestSetOfVertices(n: number, edges: number[][]): number[] {
    const map = new Map();
    for (let [from, to] of edges) {
      if (!map.has(to)) {
        map.set(to, []);
      }
      map.get(to).push([from, to]);
    }
    const ans: number[] = [];
    for (let i = 0; i < n; i++) {
      if (!map.has(i)) {
        ans.push(i);
      }
    }

    return ans;
  };

  canVisitAllRooms(rooms: number[][]): boolean {
    const seen: boolean[] = new Array(rooms.length).fill(false);
    const dfs = (room: number) => {
      if (!seen[room]) {
        seen[room] = true;
        let keys = rooms[room];
        for (const key of keys) {
          dfs(key);
        }
      }
    }

    dfs(0);

    let valid = true;
    for (const v of seen) {
      valid &&= v;
    }

    return valid;
  };

  minReorder1(n: number, connections: number[][]): number {
    let graph = new Map();
    let roads = new Map();//existing roads
    let convertToHash = (row: number, col: number) => {
      return row + "-" + col;
    }
    //build graph
    for (const [x, y] of connections) {
      if (!graph.has(x)) {
        graph.set(x, []);
      }
      if (!graph.has(y)) {
        graph.set(y, []);
      }
      graph.get(x).push(y);
      graph.get(y).push(x);
      roads.set(convertToHash(x, y), [x, y]);
    }

    let seen: boolean[] = [];

    let count = 0;
    const travel = (curr: number) => {
      seen[curr] = true;

      //find connected cities
      for (const neighbor of graph.get(curr)) {
        console.log(`curr = ${curr}, neighbor = ${neighbor}`);

        if (!seen[neighbor]) {
          // if(roads.has(convertToHash(curr, neighbor))) {
          //   count++;
          // }
          if (roads.has(convertToHash(curr, neighbor))) {
            let [x, y] = roads.get(convertToHash(curr, neighbor));
            if (y != curr) {//starting to ending city
              count++;
            }
          }
          seen[curr] = true;
          travel(neighbor);
        }
      }
    }
    seen[0] = true;
    travel(0);
    return count;
  }


  minReorder(n: number, connections: number[][]): number {
    const graph = new Map();
    let convertToHash = (row: number, col: number) => {
      return row + "-" + col;
    }
    let roads = new Set();//existing roads

    //build graph
    for (const [x, y] of connections) {
      //find connected cities
      if (!graph.has(x)) {
        graph.set(x, []);
      }
      if (!graph.has(y)) {
        graph.set(y, []);
      }
      graph.get(x).push(y);
      graph.get(y).push(x);
      roads.add(convertToHash(x, y));
    }

    let seen: boolean[] = new Array(n).fill(false);

    let count = 0;
    const travel = (curr: number) => {
      seen[curr] = true;
      //find connected cities
      for (const neighbor of graph.get(curr)) {
        if (!seen[neighbor]) {
          if (roads.has(convertToHash(curr, neighbor))) {
            count++;
          }
          seen[neighbor] = true;
          travel(neighbor);
        }
      }
    }
    seen[0] = true;
    travel(0);
    return count;
  }

  numIslands1(grid: string[][]): number {
    let graph = new Map();
    let numOfRows = grid.length;
    let numOfCols = grid[0].length;
    //init graph
    for (let i = 0; i < numOfRows; i++) {
      for (let j = 0; j < numOfCols; j++) {
        graph.set(`${i}${j}`, []);
      }
    }

    let ans = 0;
    let seen = new Map();//key is 'row col'
    const dfs = (node: string) => {
      for (const neighbor of graph.get(node)) {
        let temp = neighbor.split('');
        let i = parseInt(temp[0]);
        let j = parseInt(temp[1]);
        if (!seen.get(neighbor) && grid[i][j] == "1") {
          seen.set(neighbor, true);
          dfs(neighbor);
        }
      }
    }

    //build graph
    for (let i = 0; i < numOfRows; i++) {
      for (let j = 0; j < numOfCols; j++) {
        let val = parseInt(grid[i][j]);
        if (val == 1) {
          //if current is land, find all neighbors, up/down/left/right
          let up = i >= 1 ? parseInt(grid[i - 1][j]) : 0;
          let down = i <= numOfRows - 2 ? parseInt(grid[i + 1][j]) : 0;
          let left = j >= 1 ? parseInt(grid[i][j - 1]) : 0;
          let right = j <= numOfCols - 2 ? parseInt(grid[i][j + 1]) : 0;
          if (up == 1) {
            graph.get(`${i}${j}`).push(`${i - 1}${j}`);
          }
          if (down == 1) {
            graph.get(`${i}${j}`).push(`${i + 1}${j}`);
          }
          if (left == 1) {
            graph.get(`${i}${j}`).push(`${i}${j - 1}`);
          }
          if (right == 1) {
            graph.get(`${i}${j}`).push(`${i}${j + 1}`);
          }
        }
      }
    }

    for (let i = 0; i < numOfRows; i++) {
      for (let j = 0; j < numOfCols; j++) {
        if (!seen.get(`${i}${j}`) && grid[i][j] == "1") {
          ans++;
          seen.set(`${i}${j}`, true);
          dfs(`${i}${j}`);
        }
      }
    }

    return ans;

  };

  findCircleNum(isConnected: number[][]) {
    let dfs = (node: number) => {
      for (const neighbor of graph.get(node)) {
        // the next 2 lines are needed to prevent cycles
        if (!seen[neighbor]) {
          seen[neighbor] = true;
          dfs(neighbor);
        }
      }
    }

    // build the graph
    let n = isConnected.length;
    let graph = new Map();
    for (let i = 0; i < n; i++) {
      graph.set(i, []);
    }

    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (isConnected[i][j] == 1) {
          graph.get(i).push(j);
          graph.get(j).push(i);
        }
      }
    }

    let seen = new Array(n).fill(false);
    let ans = 0;

    for (let i = 0; i < n; i++) {
      if (!seen[i]) {
        ans++;
        seen[i] = true;
        dfs(i);
      }
    }

    return ans;
  };

  isInterleave(s1: string, s2: string, s3: string): boolean {
    let result: string[] = [];
    let s1Array: string[] = s1.split('');
    let s2Array: string[] = s2.split('');
    let s3Array: string[] = s3.split('');
    for (let i = 0; i < s3Array.length; i++) {
      let c = s3Array[i];
      if (s1Array.length && c == s1Array[0]) {
        result.push(s1Array.shift()!);
      }
      if (s2Array.length && c == s2Array[0]) {
        result.push(s2Array.shift()!);
      } else {

      }
    }

    const r = result.join('');
    return r == s3 && s1Array.length == 0 && s2Array.length == 0;
  };

  findNode(node: TreeNode | null, val: number): TreeNode | null {
    if (!node) {
      return null;
    }

    if (node.val == val) {
      return node;
    }

    let left = this.findNode(node.left, val);
    let right = this.findNode(node.right, val);
    if (left) {
      return left;
    }
    return right;
  }

  findNodeBst(node: TreeNode | null, val: number): TreeNode | null {
    if (!node) {
      return null;
    }

    if (node.val > val) {
      if (node.left) {
        if (node.left.val < val) {
          return node;
        } else {
          return this.findNode(node.left, val);
        }
      } else {
        return node;
      }
    } else {
      if (node.right) {
        if (node.right.val > val) {
          return node;
        } else {
          return this.findNode(node.right, val);
        }
      } else {
        return node;
      }
    }

  }

  insertIntoBST(root: TreeNode | null, val: number): TreeNode | null {
    let newRoot = new TreeNode(val);

    if (!root) {
      return newRoot;
    }


    if (val < root.val) {//go left
      if (root.left) {
        this.insertIntoBST(root.left, val);
      } else {
        root.left = new TreeNode(val);
      }
    } else {
      if (root.right) {
        this.insertIntoBST(root.right, val);
      } else {
        root.right = new TreeNode(val);
      }
    }

    return root;
  };

  insertIntoBST1(root: TreeNode | null, val: number): TreeNode | null {
    if (!root) {
      return new TreeNode(val);
    }

    if (val < root.val) {//go left
      if (root.left) {
        this.insertIntoBST(root.left, val);
      } else {
        root.left = new TreeNode(val);
      }
    } else {
      if (root.right) {
        this.insertIntoBST(root.right, val);
      } else {
        root.right = new TreeNode(val);
      }
    }

    return root;
  };

  buildTree(input: (number | null)[], index: number, total: number): TreeNode | null {
    const length: number = input.length;
    if (length == 0) {
      return null;
    }

    let root: TreeNode | null = null;
    if (index < total && input[index] != null) { //item within range and not null
      root = new TreeNode(input[index]!);
      root.left = this.buildTree(input, 2 * index + 1, length);
      root.right = this.buildTree(input, 2 * index + 2, length);
    }
    return root;
  }


  buildTree1(input: (number | null)[], i: number, n: number): TreeNode | null {
    if (!input.length) {
      return null;
    }
    let root = null;
    if (i < n && input[i] != null) {
      root = new TreeNode(input[i]!);
      root.left = this.buildTree1(input, 2 * i + 1, n);
      root.right = this.buildTree1(input, 2 * i + 2, n);
    }
    return root;
  }

  isValidBST(root: TreeNode | null): boolean {
    const isValid = (node: TreeNode | null, min: number, max: number): boolean => {
      if (!node) {
        return true;
      }
      if (node.val <= min || node.val >= max) {
        return false;
      }
      let isLeftValid = true;
      let isRightValid = true;
      if (node.left) {
        isLeftValid = isValid(node.left, min, node.val);
      }
      if (node.right) {
        isRightValid = isValid(node.right, min, node.val);
      }
      return isLeftValid && isRightValid;
    }

    return isValid(root, Number.MIN_SAFE_INTEGER, Number.MAX_SAFE_INTEGER);
  }

  isValidBST1(root: TreeNode | null): boolean {
    let isValid = true;
    let array: number[] = [];
    const bst = (node: TreeNode | null) => {
      if (!node) {
        return;
      }
      if (node.left) {
        bst(node.left);
      }
      console.log(node.val);
      array.push(node.val);
      if (node.right) {
        bst(node.right);
      }
    }
    bst(root);
    for (let i = 0; i < array.length - 1; i++) {

      if (array[i + 1] <= array[i]) {
        return false;
      }
    }

    return true;

  };

  getMinimumDifference(root: TreeNode | null): number {
    if (!root) {
      return Number.MAX_SAFE_INTEGER;
    }

    let min: number = Number.MAX_SAFE_INTEGER;
    let array: number[] = [];
    const bst = (node: TreeNode | null) => {
      if (!node) {
        return;
      }
      if (node.left) {
        bst(node.left);
      }
      array.push(node.val);
      if (node.right) {
        bst(node.right);
      }
    };
    bst(root);
    for (let i = 0; i < array.length - 1; i++) {
      min = Math.min(min, array[i + 1] - array[i]);
    }

    return min;
  };

  rangeSumBST(root: TreeNode | null, low: number, high: number): number {

    const getSum = (node: TreeNode | null, left: number, right: number): number => {
      let nodeValue = 0;
      if (!node) {
        return 0;
      } else if (node.val >= left && node.val <= right) {
        nodeValue = node.val;
      }
      let leftTotal = 0, rightTotal = 0;
      if (node.left && node.val > left) {
        leftTotal = getSum(node.left, left, right);
      }
      if (node.right && node.val < right) {
        rightTotal = getSum(node.right, left, right);
      }

      return leftTotal + rightTotal + nodeValue;
    };
    let sum = getSum(root, low, high)
    return sum;
  };

  getMaxLength(node: TreeNode | null, max: number): number {
    if (!node) {
      return max;
    }
    if (!node.left && !node.right) {
      return max + 1;
    }

    const l = this.getMaxLength(node.left, max + 1);
    const r = this.getMaxLength(node.right, max + 1);
    return Math.max(l, r);
  }

  diameterOfBinaryTree(root: TreeNode | null): number {
    if (!root) {
      return 0;
    }

    const l = this.getMaxLength(root.left, 0);
    const r = this.getMaxLength(root.right, 0);
    let currMax = l + r;

    const lMax = this.diameterOfBinaryTree(root.left);
    const rMax = this.diameterOfBinaryTree(root.right);

    return Math.max(lMax, rMax, currMax);
  };

  findMinMax = (node: TreeNode | null, max: number, min: number): number => {
    if (node == null) {
      return max - min;
    }

    max = Math.max(node.val, max);
    min = Math.min(node.val, min);

    const leftDiff = this.findMinMax(node.left, max, min);
    const rightDiff = this.findMinMax(node.right, max, min);

    return Math.max(leftDiff, rightDiff);
  }

  maxAncestorDiff(root: TreeNode | null): number {
    if (root == null) {
      return 0;
    }

    const diff = this.findMinMax(root, root.val, root.val);
    return diff;
  };

  robotWithString(s: string): string {
    const sArray: string[] = s.split('');
    const tStack: string[] = [];
    const minValues: string[] = new Array(s.length).fill('z');//
    minValues[s.length - 1] = sArray[s.length - 1];
    const p: string[] = [];
    for (let i = sArray.length - 2; i >= 0; i--) {
      minValues[i] = sArray[i] < minValues[i + 1] ? sArray[i] : minValues[i + 1];
    }

    for (let i = 0; i < sArray.length; i++) {
      tStack.push(sArray[i]);
      while (tStack.length && tStack[tStack.length - 1] <= minValues[i + 1]) {
        p.push(tStack.pop()!);
      }
    }

    while (tStack.length > 0) {
      p.push(tStack.pop()!);
    }

    return p.join('');
  };

  nextGreaterElement(nums1: number[], nums2: number[]): number[] {
    const map = new Map();
    const stack: number[] = [];
    const ans = [];
    for (let i = 0; i < nums2.length; i++) {
      const n = nums2[i];
      while (stack.length > 0 && n > stack[stack.length - 1]) {
        map.set(stack.pop(), n);
      }
      stack.push(n);
    }

    while (stack.length > 0) {
      map.set(stack.pop(), -1);
    }

    for (let i = 0; i < nums1.length; i++) {
      let n = nums1[i];
      let val = map.get(n);
      ans.push(val);
    }

    return ans;
  };

  longestSubarray(nums: number[], limit: number): number {
    let increase: number[] = [];
    let decrease: number[] = [];
    let max: number = 0, l = 0;
    for (let r = 0; r < nums.length; r++) {

      //constrain the size of increase and decrease
      while (increase.length && increase[increase.length - 1] > nums[r]) {
        increase.shift();
      }
      while (decrease.length && decrease[decrease.length - 1] < nums[r]) {
        decrease.shift();
      }
      increase.push(nums[r]);
      decrease.push(nums[r]);

      //CLEAN UP INVALID CASES
      while (decrease[0] - increase[0] > limit) {
        if (nums[l] == decrease[0]) {
          decrease.shift();
        }
        if (nums[l] == increase[0]) {
          increase.shift();
        }
        l++;
      }

      max = Math.max(r - l + 1, max);
    }
    return max;
  };

  maxSlidingWindow(nums: number[], k: number): number[] {
    const queue: number[] = [0];
    const ans: number[] = [];

    if (k == 1) {
      return nums;
    }
    for (let i = 1; i < nums.length; i++) {

      //make sure we only have k size queue
      let firstIndex: number = queue[0];
      if (firstIndex < (i - k + 1)) {
        queue.shift();
      }

      let curr = nums[i];
      let lastIndex = queue[queue.length - 1];
      while (curr > nums[lastIndex] && queue.length > 0) {
        queue.pop();
        lastIndex = queue[queue.length - 1];
      }
      queue.push(i);//store index in the queue


      if (i >= k - 1) {
        let maxIndex = queue[0];
        ans.push(nums[maxIndex]);
      }
    }
    return ans;
  };

  makeGood(s: string): string {
    let stack: any[] = [];
    for (let i = 0; i < s.length; i++) {
      const c = s.charAt(i);
      const peek = stack[stack.length - 1];
      if (stack.length > 0 && peek != c && (peek == c.toLowerCase() || peek == c.toUpperCase())) {
        stack.pop();
      } else {
        stack.push(c);
      }
    }

    return stack.join('');
  };

  simplifyPath(path: string): string {
    let stack = [];
    for (let i = 0; i < path.length; i++) {
      const c = path.charAt(i);
      stack.push(c);
      if (c == '/') {
        if (i == path.length - 1) {
          stack.pop();
          break;
        }
        //get next segment
        let temp = [];
        for (let j = i + 1; j < path.length; j++) {
          const t = path.charAt(j);
          if (t != '/') {
            temp.push(t);
          } else {
            break;
          }
        }
        const next = temp.join('');
        if (next === '.') {
          stack.pop();
          i = i + 1;
        } else if (next == '..') {
          //pop previous section
          stack.pop();// pop '/'
          let count = i;
          while (count >= 0) {
            const s = stack[stack.length - 1];
            if (s != '/') {
              stack.pop();
              count--;
            } else {
              stack.pop();
              break;
            }
          }
          i = i + 2;
        } else if (next == '') {
          stack.pop();//skip '/'
        } else {
          stack.push(next);//push the valid segment
          i = i + next.length;
        }

      }
    }

    return stack.join('') || '/';
  };

  //before the node with index left
  reverse(beforeFirst: ListNode | null, left: number, right: number): ListNode | null {
    console.log(beforeFirst?.val, left, right);
    let last: any = null,
      first: ListNode | null = null,
      afterFirst: ListNode | null | undefined = null,
      beforeLast: ListNode | null = null,
      afterLast: ListNode | null = null;

    if (beforeFirst != null) {
      first = beforeFirst.next;
    }

    if (left == right) {
      return first;
    } else if (left + 1 == right) { //last step, breaking recursion
      if (first != null) {
        last = first.next;
      }
      afterFirst = first//subListStart
      beforeLast = last;//subListEnd
    } else {
      //keep running recursion
      beforeLast = this.reverse(first, left + 1, right - 1);
      ;//subListEnd
      afterFirst = first?.next;//subListStart
      last = beforeLast?.next;
    }
    //swap

    afterLast = last.next;

    if (beforeFirst != null) {
      beforeFirst.next = last;
    }
    last.next = afterFirst;
    if (beforeLast != null) {
      beforeLast.next = first;
    }
    if (first != null) {
      first.next = afterLast;
    }

    return first;
  }

  reverseBetween(head: ListNode | null, left: number, right: number): ListNode | null {
    let zeroNode = new ListNode(0, head);
    let beforeFirst: ListNode | null = zeroNode;
    for (let i = 1; i < left; i++) {
      if (beforeFirst != null) {
        beforeFirst = beforeFirst.next;
      }
    }

    this.reverse(beforeFirst, left, right);

    let test = zeroNode;
    while (test != null) {
      console.log(test.val);
      if (test.next) {
        test = test.next;
      } else {
        break;
      }
    }
    return zeroNode.next;
  }

  reverseBetween1(head: ListNode | null, left: number, right: number): ListNode | null {
    let i2 = left;
    let curr = head, prev = null;
    let startP1 = head, beforeStartNode = null, next = null;

    for (let i = 1; i < left; i++) {
      beforeStartNode = curr;
      if (curr != null) {
        startP1 = curr.next;
        curr = curr.next;//move p1
      }
    }

    while (i2 <= right && curr != null) {
      next = curr.next;//keep reference
      curr.next = prev;
      prev = curr;
      curr = next;
      i2++;
      if (i2 > right) {
        if (beforeStartNode) {
          beforeStartNode.next = prev;
        } else {
          head = prev;
        }
      }
    }
    if (startP1 != null) {
      startP1.next = curr;//relink the prev Node to current reversal
    }
    return head;
  }

  middleNode(head: ListNode | null): ListNode | null {
    if (head == null || head.next == null) {
      return head;
    }
    let p = head;
    while (p?.next != null) {
      p = p.next;
      while (p.next?.val == p.val) {
        p.next = p.next.next;
      }
    }

    return head;
  };

  containsDuplicate(nums: number[]): boolean {
    const map = new Map();
    let isFalse = 1;
    for (const num of nums) {
      map.set(num, (map.get(num) || 0) + 1);
      isFalse = isFalse * map.get(num);
    }

    return isFalse != 1;
  };

  lengthOfLongestSubstring(s: string): number {
    const map = new Map();
    const strArray = s.split('');
    let l = 0, ans = 0;
    for (let r = 0; r < strArray.length; r++) {
      const c = strArray[r];
      const count = (map.get(c) || 0) + 1;
      map.set(c, count);
      if (count == 1) {
        ans = Math.max(ans, r - l + 1);
      }
      while (count > 1 && l <= r) {
        if (strArray[l] == c) {
          map.set(strArray[l], map.get(strArray[l]) - 1);
          l++;
          break;
        }
        map.set(strArray[l], map.get(strArray[l]) - 1);
        l++;
      }
    }

    return ans;
  };

  numJewelsInStones(jewels: string, stones: string): number {
    const map = new Map();
    for (const c of jewels) {
      map.set(c, 0);
    }
    for (const c of stones) {
      if (map.has(c)) {
        map.set(c, map.get(c) + 1);
      }
    }
    let ans = 0;
    for (const c of map.values()) {
      ans += c;
    }
    return ans;
  };

  canConstruct(ransomNote: string, magazine: string): boolean {
    const map = new Map();
    for (const c of magazine) {
      map.set(c, (map.get(c) || 0) + 1)
    }
    for (const c of ransomNote) {
      if (map.has(c)) {
        let val = map.get(c);
        val--;
        if (val < 0) {
          return false;
        }
        map.set(c, val);
      } else {
        return false;
      }
    }
    return true;
  };

  findMaxLength(nums: number[]): number {
    const map = new Map();
    let max = 0, count = 0;
    map.set(0, -1);
    for (let i = 0; i < nums.length; i++) {
      if (nums[i] == 0) {
        count--;
      } else {
        count++;
      }
      if (map.has(count)) {
        max = Math.max(max, i - map.get(count));
      } else {
        map.set(count, i);
      }
    }
    return max;
  };

  numberOfSubarrays(nums: number[], k: number): number {
    const oddNums = new Map();
    let ans = 0, count = 0, curr = 0, r = 0, gap = 0, oddIndex = 0;
    oddNums.set(0, -1);
    for (r = 0; r < nums.length; r++) {
      curr += nums[r] % 2;
      if (nums[r] % 2 == 1) {
        count++;
        if (count == 1) {
          oddIndex = 1;
        }
        oddNums.set(curr, r);
      }
      if (count == k) {
        gap = oddNums.get(oddIndex) - oddNums.get(oddIndex - 1);
        ans += gap;
        gap = 0;
      }
      if (count == k + 1) {
        oddIndex++;
        count--;
        if (count == k) {
          gap = oddNums.get(oddIndex) - oddNums.get(oddIndex - 1);
          ans += gap;
          gap = 0;
        }
      }

    }

    return ans;
  };

  numberOfSubarrays1(nums: number[], k: number): number {
    let counts = new Map();
    counts.set(0, 1);// [] sum is 0, occurrence is 1
    let ans = 0, curr = 0;

    for (const num of nums) {
      curr += num % 2;
      ans += counts.get(curr - k) || 0;
      counts.set(curr, (counts.get(curr) || 0) + 1);
    }

    return ans;
  };

  subarraySum(nums: number[], k: number): number {
    let map = new Map();
    let ans = 0, curr = 0;
    map.set(0, 1);
    for (let i = 0; i < nums.length; i++) {
      curr += nums[i];
      let key = curr - k;
      let value = map.get(key);
      ans += value || 0;
      map.set(curr, (map.get(curr) || 0) + 1);
    }
    return ans;
  };

  countElements(arr: number[]): number {
    let set = new Set();
    let count = 0;
    for (let i = 0; i < arr.length; i++) {
      set.add(arr[i]);
    }

    let sumKey = 0
    for (let i = 0; i < arr.length; i++) {
      sumKey = arr[i] + 1;
      if (set.has(sumKey)) {
        count++;
      }
    }
    return count;
  };

  missingNumber(nums: number[]): number {
    let map = new Set();
    for (let i = 0; i < nums.length; i++) {
      map.add(nums[i]);
    }
    for (let i = 0; i < nums.length; i++) {
      if (!map.has(i)) {
        return i;
      }
    }
    return -1;
  };

  checkIfPangram(sentence: string): boolean {
    const letters = 'abcdefghijklmnopqrstuvwxyz';
    const map = new Map();
    for (let i = 0; i < letters.length; i++) {
      map.set(letters.charAt(i), i);
    }
    for (let i = 0; i < sentence.length; i++) {
      if (map.has(sentence.charAt(i))) {
        map.delete(sentence.charAt(i));
        if (map.size == 0) {
          return true;
        }
      }
    }
    return map.size == 0;

  };

  repeatedCharacter(s: string): string {
    const map = new Map();
    let minIndex = Number.MAX_SAFE_INTEGER;
    let ans = '';
    for (let i = 0; i < s.length; i++) {
      if (!map.has(s.charAt(i))) {
        map.set(s.charAt(i), {index: i, count: 1});
      } else {
        let temp = map.get(s.charAt(i));
        if (temp.count < 2) {
          temp.count++;
          temp.index = i;
          map.set(s.charAt(i), temp);
          minIndex = Math.min(minIndex, i);
          ans = s.charAt(minIndex);
        }
      }
    }
    return ans;
  };

  equalSubstring(s: string, t: string, maxCost: number): number {
    let l = 0, r = 0, max = 0, cost = 0;
    for (r = 0; r < s.length; r++) {
      cost += Math.abs(s.charCodeAt(r) - t.charCodeAt(r));
      if (cost <= maxCost) {
        max = Math.max(max, r - l + 1);
      }
      while (cost > maxCost && l <= r) {
        l++;
        cost -= Math.abs(s.charCodeAt(l - 1) - t.charCodeAt(l - 1))
      }
    }
    return max;
  };

  maxVowels(s: string, k: number): number {
    const vowels = new Map([
      ['a', 0],
      ['e', 1],
      ['i', 2],
      ['o', 3],
      ['u', 4],
    ]);
    let l = 0, r = 0, count = 0, max = 0, length = s.length;
    for (r = 0; r < length; r++) {
      if ((r - l + 1) <= k) {
        if (vowels.has(s.charAt(r))) {
          count++;
          max = Math.max(max, count);
        }
      } else {
        l++;
        if (vowels.has(s.charAt(l - 1))) {
          count--;
        }
        if (vowels.has(s.charAt(r))) {
          count++;
          max = Math.max(max, count);
        }
      }
    }

    return max;
  }

  minSubArrayLen(target: number, nums: number[]): number {
    let left = 0, sum = 0, min = Number.MAX_SAFE_INTEGER;

    for (let right = 0; right < nums.length; right++) {
      sum += nums[right];
      while (sum >= target) {
        min = Math.min(min, right - left + 1);
        left++;
        sum -= nums[left - 1];
      }
    }

    return min;
  }

  minSubArrayLenBinary(target: number, nums: number[]): number {
    let left = 0, right = 0, sum = 0, prefix = [nums[0]], min = 0, curr = -1, mid = 0, size = 0;

    for (let i = 1; i < nums.length; i++) {
      prefix[i] = prefix[i - 1] + nums[i];
      if (nums[i] >= target) {
        return 1;
      }
      if (prefix[i] >= target) {
        curr = curr < 0 ? i : curr; // initial position, the smallest sum bigger than target
      }
    }
    if (prefix[0] >= target) {
      return 1;
    }
    if (prefix[nums.length - 1] < target) {
      return 0;
    }
    while (curr < nums.length) {
      right = curr;
      left = 0;
      //calculate window sum
      while (left < right) {
        mid = left + Math.floor((right - left) / 2);
        sum = mid >= 1 ? prefix[curr] - prefix[mid - 1] : prefix[curr];

        if (sum >= target) {
          size = curr - mid + 1;
          min = min == 0 ? size : Math.min(min, size);
          //console.log(`left = ${left}, right = ${curr}, size = ${size}, min = ${min}`);
          left = mid + 1;
        } else {
          right = mid - 1;
        }
      }
      //move window by 1
      curr++;
    }
    return min;

  };


}

class NumArray {
  constructor(nums: number[]) {

  }

  sumRange(left: number, right: number): number {
    return 0;
  }
}


class TreeNode {
  val: number
  left: TreeNode | null
  right: TreeNode | null

  constructor(val?: number, left?: TreeNode | null, right?: TreeNode | null) {
    this.val = (val === undefined ? 0 : val)
    this.left = (left === undefined ? null : left)
    this.right = (right === undefined ? null : right)
  }
}


class ListNode {
  val: number
  next: ListNode | null

  constructor(val: number, next: ListNode | null) {
    this.val = (val === undefined ? 0 : val)
    this.next = (next === undefined ? null : next)
  }
}

class MovingAverage {
  private wSize: number;
  private sum: number = 0;

  constructor(size: number) {
    this.wSize = size;
  }

  private queue: number[] = [];

  next(val: number): number {
    this.queue.push(val);
    this.sum += val;
    if (this.queue.length > this.wSize) {
      this.sum = this.sum - (this.queue.shift() as number);
    }
    return this.sum / this.queue.length;
  }
}

class StockSpanner {
  constructor() {

  }

  private stack: { price: number, ans: number }[] = [];

  next(price: number): number {
    let ans = 1;
    while (this.stack.length > 0 && price >= this.stack[this.stack.length - 1].price) {
      ans += (this.stack.pop() as { price: number, ans: number }).ans;
    }
    this.stack.push({price, ans});

    return ans;
  }
}

