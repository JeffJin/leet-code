import {Component, OnInit} from '@angular/core';
import {data1, data2} from "../data";
import {buildLinkedList, ListNode} from "../linked-list";
import {convertTree, TreeNode} from "../tree-node";
import {findUp} from "@angular/cli/src/utilities/find-up";

@Component({
  selector: 'app-leet-code',
  templateUrl: './leet-code.component.html',
  styleUrls: ['./leet-code.component.scss']
})
export class LeetCodeComponent implements OnInit {
  ngOnInit(): void {
    // console.log('uniquePaths',  this.uniquePaths(3, 7));
    // console.log('minPathSum',  this.minPathSum([[1,3,1],[1,5,1],[4,2,1]]));
    // console.log('minPathSum',  this.minPathSum([[1,2,3],[4,5,6]]));
    // console.log('uniquePathsWithObstacles',  this.uniquePathsWithObstacles([[0,0,0],[0,1,0],[0,0,0]]));
    // console.log('uniquePathsWithObstacles',  this.uniquePathsWithObstacles([[0,1],[0,0]]));
    // console.log('minFallingPathSum',  this.minFallingPathSum([[2,1,3],[6,5,4],[7,8,9]]));
    // console.log('equalSubstring',  this.equalSubstring('abcd', 'bcdf', 3));
    // console.log('equalSubstring',  this.equalSubstring('abcd', 'cdef', 3));
    // console.log('equalSubstring',  this.equalSubstring('abcd', 'acde', 0));
    // console.log('maxVowels',  this.maxVowels('abciiidef', 3));
    // console.log('maxVowels',  this.maxVowels('aeiou', 2));
    // console.log('maxVowels',  this.maxVowels('leetcode', 3));
    // console.log('subarraySum',  this.subarraySum([1,2,3], 3));
    // console.log('subarraySum',  this.subarraySum([-1,-1,1], 0));
    // console.log('subarraySum',  this.subarraySum([1], 0));
    // console.log('checkSubarraySum',  this.checkSubarraySum([23,2,4,6,7], 6));
    // console.log('checkSubarraySum',  this.checkSubarraySum([23,2,6,4,7], 6));
    // console.log('checkSubarraySum',  this.checkSubarraySum([23,2,6,4,7], 13));
    // console.log('checkSubarraySum',  this.checkSubarraySum([0, 0], 1));
    // console.log('checkSubarraySum',  this.checkSubarraySum([0], 1));
    // console.log('checkSubarraySum',  this.checkSubarraySum([3,0], 2));
    // console.log('checkSubarraySum',  this.checkSubarraySum([1,0,1], 2));
    // console.log('numSubarrayProductLessThanK',  this.numSubarrayProductLessThanK([10,5,2,6], 100));
    // console.log('numSubarrayProductLessThanK',  this.numSubarrayProductLessThanK([1,2,3,4,5], 1));
    // console.log('numSubarrayProductLessThanK',  this.numSubarrayProductLessThanK([57,44,92,28,66,60,37,33,52,38,29,76,8,75,22], 18));
    // console.log('minOperations',  this.minOperations([1,1,4,2,3], 5));
    // console.log('minOperations',  this.minOperations([1,1,4,2,3], 11));
    // console.log('minOperations',  this.minOperations([5,6,7,8,9], 4));
    // console.log('minOperations',  this.minOperations([3,2,20,1,1,3], 10));
    // console.log('minOperations',  this.minOperations(data2, 469632073));
    // console.log('minOperations',  this.minOperations([8828,9581,49,9818,9974,9869,9991,10000,10000,10000,9999,9993,9904,8819,1231,6309], 134365));
    // console.log('minOperations',  this.minOperations([500,1,4,2,3], 500));
    // console.log('lengthOfLongestSubstring',  this.lengthOfLongestSubstring(" "));
    // console.log('lengthOfLongestSubstring',  this.lengthOfLongestSubstring("au"));
    // console.log('checkInclusion',  this.checkInclusion('ab', 'eidbaooo'));
    // console.log('checkInclusion',  this.checkInclusion('ab', 'eidboaoo'));
    // console.log('wordPattern',  this.wordPattern('abba', 'dog cat cat dog'));
    // console.log('wordPattern',  this.wordPattern('abba', 'dog cat cat fish'));

    // n3.next = n4;
    // console.log('pairSum',  this.pairSumReverse(this.buildLinkedList([5,4,2,1])));
    let list1 = buildLinkedList([1,2,3,4,5,6,7,8,9,10,11,12]);
    let list2 = buildLinkedList([1,2,3,4,5,6,7]);
    let list3 = buildLinkedList([1,2,3]);
    let list4 = buildLinkedList([1,2,3,4,5]);
    let list5 = buildLinkedList([1]);

    // console.log('oddEvenList',  this.oddEvenList(list3));
    // console.log('oddEvenList',  this.oddEvenList(list2));
    // console.log('oddEvenList',  this.oddEvenList(list4));
    // console.log('deleteDuplicates',  this.deleteDuplicates(list3));
    // console.log('deleteDuplicates',  this.deleteDuplicates(list4));
    // console.log('longestSubarray',  this.longestSubarray([8,2,4,7], 5));
    // console.log('longestSubarray',  this.longestSubarray([4,2,10,6,1], 10));
    // console.log('longestSubarray',  this.longestSubarray([8,7,4,2,8,1,7,7], 8));
    // console.log('longestSubarray',  this.longestSubarray([10,1,2,4,7,2], 5));
    // console.log('robotWithString',  this.robotWithString('zza'));
    // console.log('finalPrices',  this.finalPrices([8,4,6,2,3]));

    const node = convertTree([8,3,10,1,6,null,14,null,null,4,7,13]);
    const node1 = convertTree([1,null,2,null,0,3]);
    const node2 = convertTree([3,9,20,null,null,15,7]);
    console.log('maxAncestorDiff',  this.maxAncestorDiff(node));
    console.log('maxAncestorDiff',  this.maxAncestorDiff(node1));
    console.log('maxAncestorDiff',  this.maxAncestorDiff(node2));
  }

  maxAncestorDiff(root: TreeNode | null): number {
    if(!root) {
      return 0;
    }
    const cache = new Map();
    const dfs = (node: TreeNode | null): number[] => {
      if(!node) {
        return [Number.MAX_SAFE_INTEGER, Number.MIN_SAFE_INTEGER];
      }
      if(cache.has(node)) {
        return cache.get(node);
      }
      let [lMin, lMax] = dfs(node.left);
      let [rMin, rMax] = dfs(node.right);
      let val = [Math.min(lMin, rMin, node.val), Math.max(lMax, rMax, node.val)];
      cache.set(node, val);
      return val;
    }

    let queue = [root];
    let maxDiff = 0;
    while(queue.length) {
      let nextQueue: TreeNode[] = [];
      while(queue.length > 0) {
        let node = queue.shift()!;
        let [min, max] = dfs(node);
        maxDiff = Math.max(maxDiff, Math.abs(min - node.val), Math.abs(max - node.val));
        if(node.left) {
          nextQueue.push(node.left);
        }
        if(node.right) {
          nextQueue.push(node.right);
        }
      }

      queue = nextQueue;
    }

    return maxDiff;
  };

  minDepth(root: TreeNode | null): number {
    if(!root) {
      return 0;
    }
    let queue = [root];
    let count = 1;

    while(queue.length) {
      let nextQueue: TreeNode[] = [];
      while(queue.length > 0) {
        let node = queue.shift()!;
        if(!node.left && !node.right) {
          return count;
        }
        if(node.left) {
          nextQueue.push(node.left);
        }
        if(node.right) {
          nextQueue.push(node.right);
        }
      }
      count++;
      queue = nextQueue;
    }
    return -1;
  };

  lowestCommonAncestor(root: TreeNode | null, p: TreeNode | null, q: TreeNode | null): TreeNode | null {
    if(!root) {
      return null;
    }

    const findNode = (node: TreeNode | null, t: TreeNode | null): boolean => {
      if(!node && !t) {
        return true;
      }
      if((!node && t) || (node && !t)) {
        return false;
      }
      if(node && t && node.val == t.val) {
        return true;
      }
      return findNode(node!.left, t) || findNode(node!.right, t);
    }

    let stack = [root];
    let curr: TreeNode | null = null;
    while (stack.length) {
      let node = stack.pop()!;
      let isValid = findNode(node, p) && findNode(node, q);
      if(isValid) {
        curr = node;
      }
      if(node?.left) {
        stack.push(node.left)
      }
      if(node?.right) {
        stack.push(node.right)
      }
    }

    return curr;

  }

  goodNodes(root: TreeNode | null): number {
    if(!root) {
      return 0;
    }
    const traverse = (node: TreeNode | null, max: number): number => {
      if(!node) {
        return 0;
      }
      let count = 0
      if (node.val >= max) {
        count = 1;
      }
      max = Math.max(max, node.val);
      return traverse(node.left, max) + traverse(node.right, max) + count;
    };
    return traverse(root, root.val);
  };

  hasPathSum(root: TreeNode | null, targetSum: number): boolean {
    if (root == null) {
      return false;
    }

    const validate = (node: TreeNode | null, remaining: number): boolean => {
      if(node == null) {
        return false;
      }
      if(!node.left && !node.right && remaining == node.val) {
        return true;
      }
      remaining -= node.val;
      let left = validate(node.left, remaining);
      let right = validate(node.right, remaining);
      return left || right;
    };

    return validate(root, targetSum);
  };

  sumSubarrayMins(arr: number[]): number {
    const MOD = 1000000007;

    let stack = [];
    let sumOfMinimums = 0;

    for (let i = 0; i <= arr.length; i++) {

      // when i reaches the array length, it is an indication that
      // all the elements have been processed, and the remaining
      // elements in the stack should now be popped out.

      while (stack.length && (i == arr.length || arr[stack[stack.length - 1]] >= arr[i])) {

        // Notice the sign ">=", This ensures that no contribution
        // is counted twice. rightBoundary takes equal or smaller
        // elements into account while leftBoundary takes only the
        // strictly smaller elements into account

        const mid: number = stack.pop()!;
        const leftBoundary: number = stack[stack.length - 1] === undefined ? -1 : stack[stack.length - 1];
        const rightBoundary = i;

        // count of subarrays where mid is the minimum element
        const count = ((mid - leftBoundary) * (rightBoundary - mid)) % MOD;
        sumOfMinimums += (count * arr[mid]) % MOD;
        sumOfMinimums %= MOD;
      }
      stack.push(i);
    }

    return sumOfMinimums;
  }

  sumSubarrayMins1(arr: number[]): number {
    const MOD = 1000000007;
    let n = arr.length;
    let sum = 0;
    for(let i = 0; i < arr.length; i++) {
      let r = i, l = i;

      //move right
      for (r = i + 1; r < arr.length; r++) {
        if(arr[r] < arr[i]) {
          break;
        }
        let stack: number[] = [arr[i]];
        while (r < n && arr[r] <= stack[stack.length - 1]) {
          stack.pop();
        }
        stack.push(arr[r]);
      }

      //move left
      for (l = i - 1; l >= 0; l--) {
        if(arr[l] <= arr[i]) {
          break;
        }
        let stack: number[] = [arr[i]];
        while (l >= 0 && arr[l] < stack[stack.length - 1]) {
          stack.pop();
        }
        stack.push(arr[l]);
      }

      sum += (i - l) * (r - i) * arr[i];
    }

    return sum % MOD;
  };

  canSeePersonsCount(heights: number[]): number[] {
    let n = heights.length;
    const result = new Array<number>(n).fill(0);
    let stack: number[] = [];
    for(let i = n - 1; i >= 0; i--){
      while(stack.length > 0 && heights[i] > heights[stack[stack.length - 1]]) {
        stack.pop();
        result[i]++;
      }
      if (stack.length > 0) {
        result[i]++;
      }
      stack.push(i);
    }
    result[result.length - 1] = 0;
    return result;
  };

  finalPrices(prices: number[]): number[] {
    const stack: number[] = [];
    const result: number[] = [];
    for (let i = prices.length - 1; i >= 0; i--) {
      while(stack.length > 0 && stack[stack.length - 1] > prices[i]) {
        stack.pop();
      }
      let discount = 0;
      if(stack.length > 0) {
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
    for(let i = 0; i < senate.length; i++) {
      if(senate[i] == 'R') {
        rQueue.push(i);
      } else {
        dQueue.push(i);
      }
    }
    let rIndex = 0, dIndex = 0;
    const size = senate.length;
    while(rQueue.length > 0 && dQueue.length > 0) {
      rIndex = rQueue.shift()!;
      dIndex = dQueue.shift()!;
      if(rIndex < dIndex) {
        rQueue.push(rIndex + size);
      } else {
        dQueue.push(dIndex + size);
      }
    }

    if(rQueue.length == 0) {
      return 'Dire';
    } else {
      return 'Radiant';
    }
  };

  asteroidCollision(asteroids: number[]): number[] {
    const stack: number[] = [];
    for(let i = 0; i < asteroids.length; i++) {
      stack.push(asteroids[i]);

      if(stack.length > 1) {
        let top = stack[stack.length - 1];
        let secondTop = stack[stack.length - 2];
        while(stack.length > 1 && secondTop > 0 && top < 0){
          if(secondTop > Math.abs(top)){
            stack.pop();
          } else if(secondTop < Math.abs(top)) {
            stack.pop();
            stack.pop();
            stack.push(top);
          } else {
            //same mass, pop both
            stack.pop();
            stack.pop();
          }
          if(stack.length <= 1) {
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
    while(pushed.length > 0 && popped.length > 0) {
      temp.push(pushed.shift());

      while(popped.length > 0 && temp[temp.length - 1] == popped[0]) {
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
    for(let i = sArray.length - 1; i >= 0; i--) {
      min = sArray[i] <= min ? sArray[i] : min;
      minValues[i] = min;
    }

    for (let i = 0; i < sArray.length; i++) {
      while(tStack.length && tStack[tStack.length - 1] <= minValues[i]) {
        p.push(tStack.pop()!);
      }
      tStack.push(sArray[i]);
    }

    while(tStack.length > 0) {
      p.push(tStack.pop()!);
    }

    return p.join('');
  };


  nextGreaterElement(nums1: number[], nums2: number[]): number[] {
    const map = new Map();
    const stack = [nums2[0]];
    for (let i = 1; i < nums2.length; i++) {
      while(stack.length > 0 && nums2[i] > stack[stack.length - 1]) {
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
      while(queue1.length > 0 && curr < nums[queue1[queue1.length - 1]]) {
        queue1.pop();
      }
      queue1.push(right);


      while(queue2.length > 0 && curr > nums[queue2[queue2.length - 1]]) {
        queue2.pop();
      }
      queue2.push(right);

      //when condition violates the rule, move left by incrementing
      while(nums[queue2[0]] - nums[queue1[0]] > limit) {
        while(queue2[0] <= left) {
          queue2.shift();
        }

        while(queue1[0] <= left) {
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
    if(k == 1){
      return nums;
    }
    for (let i = 1; i < nums.length; i++) { //monotonic decreasing
      if(queue[0] < i - k + 1){
        queue.shift();
      }

      while(queue.length > 0 && nums[i] > nums[queue[queue.length - 1]]) {
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
    for(let i = 0; i < n; i++){
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
      if(arr[i] == '..') {
        if(stack.length > 0) {
          stack.pop();
        }
      }
      else if(arr[i] != '.' && arr[i] != '') {
        stack.push(arr[i]);
      }
    }
    return '/' + stack.join('/');
  };

  removeDuplicates(s: string): string {
    const stack: string[] = [];
    for(const n of s) {
      if(stack.length && stack[stack.length - 1] == n) {
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
    for(const n of s) {
      let peek = stack.length > 0 ? stack[stack.length - 1] : null;
      let val = map[n];
      if(peek && val && peek == val) {
        stack.pop();
      } else {
        stack.push(n);
      }
    }

    return stack.length == 0;
  };

  oddEvenList(head: ListNode | null): ListNode | null {
    if(!head || !head.next) {
      return head;
    }
    let arr: ListNode[] = [];
    let dummy = new ListNode(-1);
    dummy.next = head;
    let curr = head;
    let prev = dummy;
    let end = null;
    let i = 1;
    while(curr) {
      if(i % 2 == 0) {
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
    for(let i = 1; i < arr.length; i++) {
      arr[i-1].next = arr[i];
      if(i == arr.length - 1) {
        arr[i].next = null;
      }
    }

    return dummy.next;
  };

  getDecimalValue(head: ListNode | null): number {
    if(!head) {
      return 0;
    }

    const getVal = (node: ListNode | null): {pow: number, sum: number} => {
      if(!node) {
        return {pow: 0, sum: 0};
      }

      let {pow, sum} = getVal(node.next);
      if(node.val == 1) {
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
    while(curr) {
      arr.push(curr.val);
      curr = curr.next;
    }
    let n = arr.length;
    let i = 0;
    let newHead = head;
    for(let g = 1; g < n; g++) {
      let start = i;
      let end = Math.min(i + g - 1, n - 1);
      let size = end - start + 1;
      for(let idx = end; idx >= start; idx--) {
        if((size % 2) == 0) {
          //if event group size, reverse
          newHead!.val = arr[idx];
        }
        newHead = newHead!.next;
      }
      i = end + 1;
      if(i >= n) {
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
    while(curr) {
      arr.push(curr.val);
      curr = curr.next;
    }
    let n = arr.length;
    let i = 0;
    let newHead = head;
    for(let g = 1; g < n; g++) {
      let start = i;
      let end = Math.min(i + g - 1, n - 1);
      let size = end - start + 1;
      for(let idx = end; idx >= start; idx--) {
        if((size % 2) == 0) {
          //if event group size, reverse
          newHead!.val = arr[idx];
        }
        newHead = newHead!.next;
      }
      i = end + 1;
      if(i >= n) {
        break;
      }
    }

    return dummy.next;
  };

  isPalindrome(head: ListNode | null): boolean {
    if(!head){
      return false;
    }

    let dummy = new ListNode(-1);
    dummy.next = head;
    let slow = head;
    let fast = head;
    let part1 = [];
    while(fast && fast.next) {
      part1.push(slow.val);
      fast = fast.next.next!;
      slow = slow.next!;
      if(fast && !fast.next) {
        slow = slow.next!;
      }
    }
    while(part1.length && slow) {
      let val1 = part1.pop();
      let val2 = slow.val;
      slow = slow.next!;
      if(val1 !== val2) {
        return false;
      }
    }
    return true;
  };

  swapNodes(head: ListNode | null, k: number): ListNode | null {
    if(!head || !head.next) {
      return head;
    }
    let dummy = new ListNode(-1);
    dummy.next = head;

    let first: ListNode | null = head;
    let last: ListNode | null = head;

    let slow: ListNode | null = head, fast: ListNode | null = head;
    for(let i = 1; i < k; i ++) {
      fast = fast!.next;
      first = fast;
    }

    while(fast && fast.next) {
      slow = slow!.next;
      last = slow;
      fast = fast.next;
    }

    if(first == last) {
      return dummy.next;
    }

    let temp = first!.val;
    first!.val = last!.val;
    last!.val = temp;

    return dummy.next;
  };

  deleteDuplicates(head: ListNode | null): ListNode | null {
    if(!head || !head.next) {
      return head;
    }
    let dummy = new ListNode(111);
    dummy.next = head;
    let prev = dummy;
    let curr: ListNode | null = head;
    let currVal: number;
    while(curr) {
      let next: ListNode | null = curr.next;
      currVal = curr.val;
      if(next && next.val == currVal) {
        while(curr && curr.val == currVal) {
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
    if(!head!.next) {
      return null;
    }
    let dummy = new ListNode(0);
    dummy.next = head;
    let nodeToDelete = head;
    let second = head;
    let prevFirst = null;
    while(n > 0 && second) {
      second = second.next;
      n--;
    }

    while(second) {
      second = second.next;
      prevFirst = nodeToDelete;
      nodeToDelete = nodeToDelete!.next;
    }

    if(prevFirst) {
      prevFirst!.next = nodeToDelete!.next;
    } else {
      dummy.next = nodeToDelete!.next;
    }

    return dummy.next;
  };

  deleteMiddle(head: ListNode | null): ListNode | null {
    if(!head!.next) {
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
    while(node) {
      if(leftCopy > 2) {
        node = node.next;
        leftCopy--;
      } else {
        prev1 = left == 1 ? dummy : node;
        start1 = left == 1 ? node :node.next;

        prev = prev1;
        curr = start1;
        while(size > 0 && curr) {
          let next: ListNode | null = curr.next;
          curr.next = prev;
          prev = curr;
          curr = next;
          size--;
          if(size == 1) {
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
    while(node) {
      if(leftCopy > 2) {
        node = node.next;
        leftCopy--;
      } else {
        prev1 = left == 1 ? dummy : node;
        start1 = left == 1 ? node :node.next;

        prev = prev1;
        curr = start1;
        while(size > 0 && curr) {
          let next: ListNode | null = curr.next;
          curr.next = prev;
          prev = curr;
          curr = next;
          size--;
          if(size == 1) {
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
    if(!head!.next!.next) {
      return head!.val + head!.next!.val;
    }
    let fast = head;
    let slow = head;
    while(fast && fast.next) {
      fast = fast.next.next;
      slow = slow!.next;
    }
    let max = 0;
    let part2 = slow;
    let head2 = null;
    let prev = null;
    while(part2) {
      let next = part2.next;
      part2.next = prev;
      prev = part2;
      part2 = next;
      head2 = prev;
    }
    while(head2 && head) {
      max = Math.max(max, head2!.val + head.val);
      head = head.next;
      head2 = head2.next;
    }
    return max;
  };
  pairSumLoop(head: ListNode | null): number {
    const nums = [];
    while(head) {
      nums.push(head.val);
      head = head.next;
    }
    let i = 0, j = nums.length - 1;
    let max = 0;
    while(i < j){
      max = Math.max(max, nums[i] + nums[j]);
      i++;
      j--;
    }
    return max;
  };

  swapPairs(head: ListNode | null): ListNode | null {
    const dummy = new ListNode(-1);
    dummy.next = head;
    let prevNode = dummy;
    while (head && head.next) {
      const firstNode = head;
      const secondNode = head.next;
      //this will update dummy next node only once
      prevNode.next = secondNode;
      firstNode.next = secondNode.next;
      secondNode.next = firstNode;
      //reference to dummy node next is broken after this line
      prevNode = firstNode;
      head = firstNode.next;
    }
    return dummy.next;
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
    firstNode.next = this.swapPairs(secondNode.next);
    secondNode.next = firstNode;
    // Now the head is the second node
    return secondNode;
  };

  reverseList(head: ListNode | null): ListNode | null {
    if(!head || !head.next) {
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
    if(word1.length != word2.length) {
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
    for(const k1 of map1.keys()) {
      if(!map2.has(k1)) {
        return false;
      }
    }
    let vals1 = [...map1.values()].sort();
    let vals2 = [...map2.values()].sort();
    for(let i= 0; i < vals1.length; i++) {;
      if(vals1[i] != vals2[i]) {
        return false;
      }
    }
    return true;
  };

  customSortString(order: string, s: string): string {
    const mapS = new Map();
    const mapO = new Map();
    const orderArr = order.split('');
    for(let i = 0; i < s.length; i++) {
      mapS.set(s[i], (mapS.get(s[i]) || 0) + 1);
    }
    for(let i = 0; i < order.length; i++) {
      mapO.set(order[i], i);
    }

    let result = [];
    for(let i = 0; i < orderArr.length; i++) {
      let c = orderArr[i];
      if (mapS.has(c)) {
        //place s chars in order
        let count = mapS.get(c);
        while(count > 0) {
          result.push(c);
          count--;
          if(count == 0) {
            mapS.delete(c);
          }
        }
      }
    }
    for(const [s, c] of mapS) {
      for(let i = 0; i < c; i++) {
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

    if(pArr.length !== sArr.length) {
      return false;
    }

    for(let i = 0; i < pArr.length; i++){
      let p1 = pArr[i];
      let s1 = sArr[i];
      if(!map1.has(p1) && !map2.has(s1)) {
        map1.set(p1, s1);
        map2.set(s1, p1);
      } else if(!(map1.get(p1) == s1 && map2.get(s1) == p1)) {
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

    for(let i = 0; i < sArr.length; i++){
      let s1 = sArr[i];
      let t1 = tArr[i];
      if(!map1.has(s1) && !map2.has(t1)) {
        map1.set(s1, t1);
        map2.set(t1, s1);
      } else if(!(map1.get(s1) == t1 && map2.get(t1) == s1)) {
        return false;
      }
    }

    return true;
  };

  checkInclusion(s1: string, s2: string): boolean {
    const map1 = new Map();
    for(let s of s1){
      map1.set(s, (map1.get(s) || 0) + 1);
    }

    const compare = (map1: Map<string, string>, map2: Map<string, string>): boolean => {
      for(const [k1, v1] of map1) {
        if(!map2.has(k1)) {
          return false;
        }
        let v2 = map2.get(k1)!;
        if(v1 != v2) {
          return false;
        }
      }
      return true;
    }

    let i = 0;
    const map2 = new Map();
    for(let j = 0; j < s2.length; j++) {
      let s = s2[j];
      map2.set(s, (map2.get(s) || 0) + 1);
      if((j - i + 1) == s1.length) {
        if(compare(map1, map2)) {
          return true;
        }
        //reduce s2[i] count
        let count = map2.get(s2[i]);
        count--;
        if(count <= 0) {
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
    for(let j = 0; j < nums.length; j++) {
      curr += nums[j];
      map.set(nums[j], (map.get(nums[j]) || 0) + 1);

      while(map.has(nums[j]) && map.get(nums[j]) > 1 && i <= j) {
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
    for(let j = 0; j < nums.length; j++) {
      curr += nums[j];
      if(map.has(curr - goal)) {
        count += map.get(curr - goal);
      }
       map.set(curr, (map.get(curr) || 0) + 1);
    }

    return count;
  };

  maxSubarrayLength(nums: number[], k: number): number {
    const map = new Map();
    let i = 0, max = 0;
    for(let j = 0; j < nums.length; j++) {
      let num = nums[j];
      let count = (map.get(num) || 0) + 1;
      map.set(num, count);
      if(count <= k) {
        max = Math.max(max, j - i + 1);
      }
      while(count > k) {
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
    for(const dir of path) {
      if(dir == 'N') {
        point = [point[0], point[1] + 1];
      }
      if(dir == 'S') {
        point = [point[0], point[1] - 1];
      }
      if(dir == 'E') {
        point = [point[0] + 1, point[1]];
      }
      if(dir == 'W') {
        point = [point[0] - 1, point[1]];
      }
      let key = getKey(point);
      if(set.has(key)) {
        return true;
      }
    }
    return false;
  };

  frequencySort(s: string): string {
    const map = new Map();
    for(const c of s) {
      map.set(c, (map.get(c) || 0)+1);
    }
    const temp = [...map.entries()] ;
    temp.sort(([k1, v1], [k2, v2]) => v2 - v1);
    const result: string[] = [];
    for(const [k, v] of temp) {
      for(let i = 0; i < v; i++) {
        result.push(k);
      }
    }
    return result.join('');
  };

  containsDuplicate(nums: number[]): boolean {
    const map = new Map();
    for(const num of nums) {
      map.set(num, (map.get(num) || 0)+1);
      if(map.get(num) > 1) {
        return false;
      }
    }

    return true;
  };

  lengthOfLongestSubstring(s: string): number {
    const map = new Map();
    let max = 0;
    let left = 0;
    for(let i = 0; i < s.length; i++) {
      let key = s[i];
      if(map.has(key)) {
        max = Math.max(max, i - left);
        while(left < map.get(key) + 1) {
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
    for(let i = 0; i < n; i++) {
      let row = grid[i];
      let key = getKey(row);
      map.set(key, (map.get(key) || 0) + 1);
    }

    for(let i = 0; i < n; i++) {
      let col = [];
      for(let j = 0; j < n; j++) {
        col.push(grid[j][i]);
      }
      let key = getKey(col);
      if(map.has(key)){
        count += map.get(key);
      }
    }

    return count;
  };

  maximumSum(nums: number[]): number {
    let max = -1;
    const map = new Map();

    for(let i = 0; i < nums.length; i++) {
      let strArr = (nums[i] + '').split('');
      let sum = strArr.map((c) => parseInt(c)).reduce((acc, curr) => acc + curr, 0);
      let arr = map.get(sum);
      if(!arr) {
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
    for(let i = 0; i < cards.length; i++) {
      if(!map.has(cards[i])) {
        map.set(cards[i], [i, 1]);
      } else {
        let [x, y] = map.get(cards[i]);
        map.set(cards[i], [i, y+1]);
        min = Math.min(i - x + 1, min);
      }
    }
    return min > cards.length ? -1 : min;
  };

  groupAnagrams(strs: string[]): string[][] {
    const result = [];
    const map = new Map();
    for(const s of strs) {
      let temp = s.split('').sort();
      let key = temp.join('');
      map.set(key, (map.get(key) || []).push(s));
    }

    for(const [key, values] of map) {
      result.push(values);
    }

    return result;
  };

  findWinners(matches: number[][]): number[][] {
    const loser = new Map();
    for(const [w, l] of matches) {
      loser.set(l, (loser.get(l) || 0) + 1);
    }
    const part1 = [], part2 = [];

    for(const [k, c] of loser) {
      if(loser.get(k) == 0) {
        part1.push(k);
      }
      if(c == 1) {
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
    for(let r = 0; r < n; r++) {
      if(nums[r] == 1){
        count++;
      } else {
        count--;
      }
      if(map.has(count)) {
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

    for(let i = 0; i < n; i++) {
      let odd = nums[i] % 2 == 0 ? 0 : 1;
      currOddCount += odd;
      result += map.get(currOddCount-k) || 0;
      map.set(currOddCount, (map.get(currOddCount) || 0) + 1);
    }
    return result;
  };

  getAverages(nums: number[], k: number): number[] {
    const result = [];
    const prefix = [nums[0]];
    for(let i = 1; i < nums.length; i++) {
      prefix.push(prefix[prefix.length - 1] + nums[i]);
    }

    for(let i = 0; i < nums.length; i++) {
      if (i < k || i > (nums.length - k - 1)) {
        result.push(-1);
      } else {
        let avg = (prefix[i+k] - prefix[i-k] + nums[i-k]) / (1 + 2 * k);
        result.push(Math.trunc(avg));
      }
    }
    return result;
  };

  minOperations(nums: number[], x: number): number {
    const n = nums.length;
    let prefix: number[] = [nums[0]];
    for(let i = 1; i < n; i ++) {
      prefix.push(prefix[prefix.length - 1] + nums[i]);
    }

    const reverseSum = prefix[prefix.length - 1] - x;
    if(reverseSum < 0) {
      return -1;
    } else if( reverseSum == 0) {
      return n;
    }
    let l = 0, count = -1;
    let curr = 0;
    for(let r = 0; r < n; r++) {
      curr = prefix[r] - prefix[l] + nums[l];
      while(curr > reverseSum) {
        curr = curr - nums[l];
        l++;
      }
      if(curr == reverseSum) {
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
      if(r == 0) {
        return n - arr.length;
      }
      if(r < 0 || arr.length == 0) {
        return n+1;
      }
      const key = getKey(arr, r);
      if(cache.has(key)) {
        return cache.get(key);
      }
      const removeFromLeft = arr.slice(1);
      const removeFromRight = arr.slice(0, arr.length - 1);

      const val = Math.min(dp(removeFromLeft, r - arr[0]), dp(removeFromRight, r - arr[arr.length - 1]));
      cache.set(key, val);
      return val;
    }
    const result = dp(nums, x);
    return result == n+1 ? -1 : result;
  };

  subarraysDivByK(nums: number[], k: number): number {
    const map = new Map();
    map.set(0, 1);
    let prefix = 0, count = 0;
    for (let i = 0; i < nums.length; i++) {
      prefix = prefix + nums[i] % k + k; // get positive modulo
      let mod = prefix % k ;
      if (map.has(mod)) {
        count += map.get(mod);
      }
      map.set(mod, (map.get(mod) || 0) + 1 );
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
    while(i <= j && j < nums.length) {
      prod = prod * nums[j];
      while (prod >= k && i <= j) {
        prod = prod / nums[i];
        i++;
      }
      if(prod < k) {
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
    for(let i = 0; i < nums.length; i++) {
      mod = (nums[i] + mod) % k ;
      if(map.has(mod)) {
        let left = map.get(mod);
        if(i - left > 1) {
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
    for(let i = 0; i < nums.length; i++) {
      curr += nums[i];
      if(map.has(curr - k)) {
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
    for(let i = 1; i < s.length; i++) {
      const val = vowels.has(s[i]) ? 1 : 0;
      prefix.push(prefix[prefix.length - 1] + val);
    }

    let i = 0, j = Math.min(k-1, s.length - 1), max = 0;
    let val;
    while(j < s.length) {
      val = prefix[j] - prefix[i] + (vowels.has(s[i]) ? 1 : 0);
      max = Math.max(val, max);
      j++;
      i++;
    }

    return max;
  };

  equalSubstring(s: string, t: string, maxCost: number): number {
    const prefix: number[] = [0];
    for(let i = 0; i < s.length; i++) {
      const diff = Math.abs(t.charCodeAt(i) - s.charCodeAt(i));
      if (i == 0) {
        prefix.push(diff);
      } else {
        const lastVal = prefix[prefix.length - 1];
        prefix.push(lastVal + diff)
      }
    }

    let max = 0, i = 0, j = 0;
    while(i <= j && j <= s.length) {
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
    for(let r = 0; r < nums.length; r++) {
      prod = prod * nums[r];
      while(prod >= k && left <= r) {
        prod = prod / nums[left];
        left++;
      }
      count += r - left + 1;
    }
    return count;
  };
  numSubarray(nums: number[]): number {
    let count = 0;
    for(let r = 0; r < nums.length; r++) {
      for (let l = 0; l < r; l++) {
        if(r > l) {
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
    for(let j = 0; j < n; j++) {
      if(s[j] == '0') {
        zeros++;
      }

      while(zeros > 1) {
        if (s[i] == '0') {
          zeros--;
        }
        i++;
      }
      max = Math.max(max, j - i + 1);
    }

    return max;
  }

  sortedSquares(nums: number[]): number[] {
    let minIndex = -1;
    let min = Number.MAX_SAFE_INTEGER;
    const ans = [];
    for(let i = 0; i < nums.length; i++) {
      nums[i] = nums[i] * nums[i];
      if(min > nums[i]){
        min = nums[i];
        minIndex = i;
      }
    }
    ans.push(nums[minIndex]);
    let i = minIndex - 1;
    let j = minIndex + 1;
    while(i >= 0 && j < nums.length){
      if(nums[i] > nums[j]){
        ans.push(nums[j]);
      }else{
        ans.push(nums[i]);
      }
    }
    return ans;

  };

  twoSum(nums: number[], target: number): number[] {
    const n = nums.length;
    nums.sort((a, b)=> a - b);
    let i = 0, j = n -1;

    while(i < j) {
      const temp = nums[i] + nums[j];
      if(temp == target) {
        return [i, j];
      }
      if(temp > target){
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
      if(col >= n || col < 0) {
        return Number.MAX_SAFE_INTEGER;
      }
      if(row == m - 1) {
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
    for(let i = 0; i < n; i++) {
      ans = Math.min(ans, dp(0, i))
    }
    return ans;
  };

  uniquePathsWithObstacles(obstacleGrid: number[][]): number {
    const m = obstacleGrid.length;
    const n = obstacleGrid[0].length;
    let memo: number[][] = [];
    for (let i = 0; i <= m; i++) {
      memo.push(new Array(n+1).fill(-1));
    }
    if(obstacleGrid[m-1][n-1] == 1 || obstacleGrid[0][0] == 1){
      return 0;
    }
    const dp = (row: number, col: number): number => {
      if(row == 0 && col == 0) {
        return 1;
      }
      if(row < 0 || col < 0 || obstacleGrid[row][col] == 1) {
        return 0;
      }

      if (memo[row][col] && memo[row][col] != -1) {
        return memo[row][col];
      }

      let leftNext = 0;
      if(col > 0) {
        leftNext = dp(row, col - 1);
      }
      let upNext = 0;
      if(row > 0) {
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
      memo.push(new Array(n+1).fill(-1));
    }
    if(obstacleGrid[m-1][n-1] == 1){
      return 0;
    }
    const dp = (row: number, col: number): number => {
      if(row == m - 1 && col == n - 1) {
        return 1;
      }
      if(row >= m || col >= n || obstacleGrid[row][col] == 1) {
        return 0;
      }

      if (memo[row][col] && memo[row][col] != -1) {
        return memo[row][col];
      }

      let rightNext = 0;
      if(col < n) {
        rightNext = dp(row, col + 1);
      }
      let downNext = 0;
      if(row < m) {
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
      memo.push(new Array(numOfCols+1).fill(-1));
    }

    const dp = (i: number, j: number): number => {
      if(i == numOfRows - 1 && j == numOfCols - 1) {
        return grid[i][j];
      }
      if(i >= numOfRows || j >= numOfCols) {
        return Number.MAX_SAFE_INTEGER;
      }

      if(memo[i][j] != -1){
        return memo[i][j]
      }


      let down = dp(i, j+1) + grid[i][j];
      let right = dp(i+1, j) + grid[i][j];

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
      if(row == m - 1 && col == n - 1) {
        return 1;
      }

      if (memo[row][col] && memo[row][col] != -1) {
        return memo[row][col];
      }

      let rightNext = 0;
      if(col < n) {
        rightNext = dp(row, col + 1);
      }
      let downNext = 0;
      if(row < m) {
        downNext = dp(row + 1, col);
      }

      const t = rightNext + downNext;
      memo[row][col] = t;
      return t;
    }

    return dp(0, 0);
  };

}
