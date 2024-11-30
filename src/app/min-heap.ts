export class MinHeap {
  private arr: number[] = [];

  constructor(nums: number[] = []) {
    this.arr = this.heapify(nums);
  }

  private parentIndex(childIndex: number): number {
    return Math.floor((childIndex - 1) / 2);
  }

  private leftChildIndex(parentIndex: number): number {
    return parentIndex * 2 + 1;
  }

  private rightChildIndex(parentIndex: number): number {
    return parentIndex * 2 + 2;
  }

  private heapifyDown(nums: number[], index: number) {
    let minIndex = index;
    let leftIdx = this.leftChildIndex(index);
    let rightIdx = this.rightChildIndex(index);
    if(leftIdx < nums.length && nums[leftIdx] < nums[minIndex]) {
      minIndex = leftIdx;
    }
    if(rightIdx < nums.length && nums[rightIdx] < nums[minIndex]) {
      minIndex = rightIdx;
    }
    if(minIndex != index) {
      [nums[index], nums[minIndex]] = [nums[minIndex], nums[index]];
      this.heapifyDown(nums, minIndex);
    }
  }

  heapifyUp(nums: number[], index: number)  {
    let parentIdx = this.parentIndex(index);
    if(parentIdx >= 0 && nums[parentIdx] > nums[index]) {
      [nums[index], nums[parentIdx]] = [nums[parentIdx], nums[index]];
      this.heapifyUp(nums, parentIdx);
    }
  }

  heapify(nums: number[]): number[] {
    for (let i = Math.floor(nums.length / 2) - 1; i >= 0 ; i--) {
      this.heapifyDown(nums, i);
    }
    return nums;
  }

  add(num: number): void {
    this.arr.push(num);

    this.heapifyUp(this.arr, this.arr.length - 1);
  }

  remove(): number | undefined {
    [this.arr[this.arr.length - 1], this.arr[0]] =  [this.arr[0], this.arr[this.arr.length - 1]];
    let result = this.arr.pop();
    this.heapifyDown(this.arr, 0);
    return result;
  }

  peek(): number | undefined {
    return this.arr[0];
  }

  size(): number {
    return this.arr.length;
  }

  printHeap() {
    console.log(`[${this.arr.join(', ')}]`);
  }
}
