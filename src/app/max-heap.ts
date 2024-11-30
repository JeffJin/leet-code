export class MaxHeap {
  private arr: number[] = [];

  constructor(nums: number[] = []) {
    this.arr = this.heapify(nums);
  }

  private heapifyDown(nums: number[], index: number) {
    const leftIndex = index * 2 + 1;
    const rightIndex = index * 2 + 2;
    let maxIndex = index;
    if(leftIndex < nums.length && nums[leftIndex] > nums[maxIndex]) {
      maxIndex = leftIndex;
    }
    if(rightIndex < nums.length && nums[rightIndex] > nums[maxIndex]) {
      maxIndex = rightIndex;
    }
    if(maxIndex != index) {
      [nums[maxIndex], nums[index]] = [nums[index], nums[maxIndex]];
      this.heapifyDown(nums, maxIndex);
    }
  }

  heapifyUp(nums: number[], index: number)  {
    const parentIndex = Math.floor((index - 1) / 2);
    if(parentIndex >= 0 && nums[parentIndex] < nums[index]) {
      [nums[parentIndex], nums[index]] = [nums[index], nums[parentIndex]];
      this.heapifyUp(nums, parentIndex);
    }
  }

  heapify(nums: number[]): number[] {
    for(let i = Math.floor(nums.length/2) - 1; i >= 0; i--) {
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

  print() {
    console.log(`[${this.arr.join(', ')}]`);
  }
}
