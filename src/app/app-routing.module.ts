import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import {LeetCodeComponent} from './leet-code/leet-code.component';
import {AppComponent} from "./app.component";
import {SampleComponent} from "./sample/sample.component";

const routes: Routes = [
  { path: 'sample', component: SampleComponent },
  { path: 'leetcode', component: LeetCodeComponent },
  { path: '', component: LeetCodeComponent },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
