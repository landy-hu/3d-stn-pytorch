#include <TH/TH.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#define real float


int BilinearSamplerBCHW_updateOutput(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output)
{

  int batchsize = inputImages->size[0];
  int inputImages_channels = inputImages->size[1];
  int inputImages_height = inputImages->size[2];
  int inputImages_width = inputImages->size[3];
  int inputImages_depth = inputImages->size[4];
//  printf("h %d w %d d %d  ", inputImages_height,inputImages_width,inputImages_depth);
  int output_height = output->size[2];
  int output_width = output->size[3];
  int output_depth = output->size[4];

  int output_strideBatch = output->stride[0];
  int output_strideHeight = output->stride[2];
  int output_strideWidth = output->stride[3];
  int output_strideDepth = output->stride[4];
  int output_strideChannel= output->stride[1];


  int inputImages_strideBatch = inputImages->stride[0];
  int inputImages_strideHeight = inputImages->stride[2];
  int inputImages_strideWidth = inputImages->stride[3];
  int inputImages_strideDepth = inputImages->stride[4];
  int inputImages_strideChannel = inputImages->stride[1];
//  printf("%d %d %d %d %d\n",inputImages_strideBatch,inputImages_strideHeight,inputImages_strideWidth,inputImages_strideDepth,inputImages_strideChannel);
  int grids_strideBatch = grids->stride[0];
  int grids_strideHeight = grids->stride[2];
  int grids_strideWidth = grids->stride[3];
  int grids_strideDepth = grids->stride[4];
  int grids_strideChannel = grids->stride[1];
//  printf("%d %d %d %d %d\n",grids_strideBatch,grids_strideHeight,grids_strideWidth,grids_strideDepth,grids_strideChannel);

  real *inputImages_data, *output_data, *grids_data;
  inputImages_data = THFloatTensor_data(inputImages);
  output_data = THFloatTensor_data(output);
  grids_data = THFloatTensor_data(grids);

  int b, yOut, xOut, zOut;

  for(b=0; b < batchsize; b++)
  {
    for(yOut=0; yOut < output_height; yOut++)
    {
      for(xOut=0; xOut < output_width; xOut++)
      {
        for(zOut=0; zOut<output_depth; zOut++)
        {
            real zf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth+zOut*grids_strideDepth+2*grids_strideChannel];

            real yf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth+zOut*grids_strideDepth + grids_strideChannel];
            real xf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth+zOut*grids_strideDepth ];

            // get the weights for interpolation
            int yInTopLeft, xInTopLeft,zInTopLeft;
            real yWeightTopLeft, xWeightTopLeft, zWeightTopLeft;
//            printf("zf %f  yf %f xf %f", xf, yf, zf);
            real xcoord = (xf + 1) * (inputImages_width - 1) / 2.0; //+1 ->[0,2]; *31 -> interger
            xInTopLeft = floor(xcoord); //transforming the grid coordinete into the interger coordinate
            xWeightTopLeft = 1 - (xcoord - xInTopLeft);

            real ycoord = (yf + 1) * (inputImages_height - 1) / 2.0;
            yInTopLeft = floor(ycoord);
            yWeightTopLeft = 1 - (ycoord - yInTopLeft);

            real zcoord = (zf + 1) * (inputImages_depth - 1) / 2.0;
            zInTopLeft = floor(zcoord);
            zWeightTopLeft = 1 - (zcoord - zInTopLeft);
//            printf("zf %d  yf %d xf %d\n", zInTopLeft, yInTopLeft, xInTopLeft);



            const int outAddress = output_strideBatch * b + output_strideHeight * yOut + output_strideWidth * xOut + zOut*output_strideDepth;
//            printf("outAddress %d", outAddress);
            const int inTopLeftBackAddress = inputImages_strideBatch * b + inputImages_strideHeight * yInTopLeft + inputImages_strideWidth * xInTopLeft + inputImages_strideDepth*zInTopLeft;
//            printf("   inTopLeftBackAddress %d ", inTopLeftBackAddress);
            const int inTopRightBackAddress = inTopLeftBackAddress + inputImages_strideWidth;
            const int inBottomLeftBackAddress = inTopLeftBackAddress + inputImages_strideHeight;
            const int inBottomRightBackAddress = inBottomLeftBackAddress + inputImages_strideWidth;

            const int inTopLeftForeAddress = inTopLeftBackAddress + inputImages_strideDepth;
            const int inTopRightForeAddress = inTopLeftForeAddress + inputImages_strideWidth;
            const int inBottomLeftForeAddress = inTopLeftForeAddress + inputImages_strideHeight;
            const int inBottomRightForeAddress = inBottomLeftForeAddress + inputImages_strideWidth;

            real v=0;
            real inTopLeftBack=0;
            real inTopRightBack=0;
            real inBottomLeftBack=0;
            real inBottomRightBack=0;
            real inTopLeftFore=0;
            real inTopRightFore=0;
            real inBottomLeftFore=0;
            real inBottomRightFore=0;

            // we are careful with the boundaries
            bool topLeftBackIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_width-1 && yInTopLeft >= 0 && yInTopLeft <= inputImages_height-1 && zInTopLeft>=0 && zInTopLeft <=inputImages_depth-1;
            bool topRightBackIsIn = xInTopLeft+1 >= 0 && xInTopLeft+1 <= inputImages_width-1 && yInTopLeft >= 0 && yInTopLeft <= inputImages_height-1 && zInTopLeft >=0 &&zInTopLeft<=inputImages_depth-1 ;
            bool bottomLeftBackIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_width-1 && yInTopLeft+1 >= 0 && yInTopLeft+1 <= inputImages_height-1&& zInTopLeft >=0 &&zInTopLeft<=inputImages_depth-1;
            bool bottomRightBackIsIn =xInTopLeft+1>=0&&xInTopLeft+1<= inputImages_width-1 && yInTopLeft+1 >= 0 && yInTopLeft+1<=inputImages_height-1&& zInTopLeft>=0&&zInTopLeft<=inputImages_depth-1;

            bool topLeftForeIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_width-1 && yInTopLeft >= 0 && yInTopLeft <= inputImages_height-1 && zInTopLeft+1>=0 && zInTopLeft+1 <=inputImages_depth-1;
            bool topRightForeIsIn = xInTopLeft+1 >= 0&& xInTopLeft+1<= inputImages_width-1 && yInTopLeft>=0&& yInTopLeft <= inputImages_height-1 && zInTopLeft+1 >=0 &&zInTopLeft+1<=inputImages_depth-1 ;
            bool bottomLeftForeIsIn = xInTopLeft >=0&& xInTopLeft<= inputImages_width-1 && yInTopLeft+1 >= 0 && yInTopLeft+1 <= inputImages_height-1&& zInTopLeft+1>=0 &&zInTopLeft+1<=inputImages_depth-1;
            bool bottomRightForeIsIn =xInTopLeft+1>=0&& xInTopLeft+1<=inputImages_width-1&&yInTopLeft+1>=0&& yInTopLeft+1 <= inputImages_height-1&& zInTopLeft+1 >=0 &&zInTopLeft+1<=inputImages_depth-1;

            int t;
            // interpolation happens here
            for(t=0; t<inputImages_channels; t++)
            {
//               printf("inputImages_channels %d", inputImages_channels);
               if(topLeftBackIsIn) inTopLeftBack = inputImages_data[inTopLeftBackAddress + t*inputImages_strideChannel];
//               printf("  inTopLeftBackAddress %d  inTopLeftBack %.6f  \n",inTopLeftBackAddress, inTopLeftBack);}
               if(topRightBackIsIn) inTopRightBack = inputImages_data[inTopRightBackAddress + t * inputImages_strideChannel];
               if(bottomLeftBackIsIn) inBottomLeftBack = inputImages_data[inBottomLeftBackAddress + t * inputImages_strideChannel];
               if(bottomRightBackIsIn) inBottomRightBack = inputImages_data[inBottomRightBackAddress + t * inputImages_strideChannel];

               if(topLeftForeIsIn) inTopLeftFore = inputImages_data[inTopLeftForeAddress + t * inputImages_strideChannel];
               if(topRightForeIsIn) inTopRightFore = inputImages_data[inTopRightForeAddress + t * inputImages_strideChannel];
               if(bottomLeftForeIsIn) inBottomLeftFore = inputImages_data[inBottomLeftForeAddress + t * inputImages_strideChannel];
               if(bottomRightForeIsIn) inBottomRightFore = inputImages_data[inBottomRightForeAddress + t * inputImages_strideChannel];

               v = xWeightTopLeft * yWeightTopLeft *zWeightTopLeft* inTopLeftBack
                 + (1 - xWeightTopLeft) * yWeightTopLeft *zWeightTopLeft* inTopRightBack
                 + xWeightTopLeft * (1 - yWeightTopLeft) *zWeightTopLeft* inBottomLeftBack
                 + (1 - xWeightTopLeft) * (1 - yWeightTopLeft) *zWeightTopLeft* inBottomRightBack
                 + xWeightTopLeft * yWeightTopLeft *(1-zWeightTopLeft)* inTopLeftFore
                 + (1 - xWeightTopLeft) * yWeightTopLeft *(1-zWeightTopLeft)* inTopRightFore
                 + xWeightTopLeft * (1 - yWeightTopLeft) *(1-zWeightTopLeft)* inBottomLeftFore
                 + (1 - xWeightTopLeft) * (1 - yWeightTopLeft) *(1-zWeightTopLeft)* inBottomRightFore;
//                printf("  v %.8f\n",v);
               output_data[outAddress + t * output_strideChannel] = v;
            }
        }

      }
    }
  }

  return 1;
}



int BilinearSamplerBCHW_updateGradInput(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *gradInputImages,
                                        THFloatTensor *gradGrids, THFloatTensor *gradOutput)
{

  bool onlyGrid=false;

  int batchsize = inputImages->size[0];
  int inputImages_height = inputImages->size[2];
  int inputImages_width = inputImages->size[3];
  int inputImages_depth = inputImages->size[4];
//  printf("%d\n",inputImages_depth);
  int inputImages_channels = inputImages->size[1];

  int gradOutput_height = gradOutput->size[2];
  int gradOutput_width = gradOutput->size[3];
  int gradOutput_depth = gradOutput->size[4];


  int gradOutput_strideBatch = gradOutput->stride[0];

  int gradOutput_strideHeight = gradOutput->stride[2];
  int gradOutput_strideWidth = gradOutput->stride[3];
  int gradOutput_strideDepth = gradOutput->stride[4];
  int gradOutput_strideChannel = gradOutput->stride[1];

  int inputImages_strideBatch = inputImages->stride[0];
  int inputImages_strideHeight = inputImages->stride[2];
  int inputImages_strideWidth = inputImages->stride[3];
  int inputImages_strideDepth = inputImages->stride[4];
  int inputImages_strideChannel = inputImages->stride[1];
//  printf("%d\n",inputImages_strideChannel);


  int gradInputImages_strideBatch = gradInputImages->stride[0];
  int gradInputImages_strideHeight = gradInputImages->stride[2];
  int gradInputImages_strideWidth = gradInputImages->stride[3];
  int gradInputImages_strideDepth = gradInputImages->stride[4];
  int gradInputImages_strideChannel = gradInputImages->stride[1];

  int grids_strideBatch = grids->stride[0];
//   printf("%d\n",grids_strideBatch);

  int grids_strideHeight = grids->stride[2];
  int grids_strideWidth = grids->stride[3];
  int grids_strideDepth = grids->stride[4];
  int grids_strideChannel = grids->stride[1];
//  printf("%d\n",grids_strideChannel);

  int gradGrids_strideBatch = gradGrids->stride[0];
  int gradGrids_strideHeight = gradGrids->stride[2];
  int gradGrids_strideWidth = gradGrids->stride[3];
  int gradGrids_strideDepth = gradGrids->stride[4];
  int gradGrids_strideChannel = gradGrids->stride[1];

  real *inputImages_data, *gradOutput_data, *grids_data, *gradGrids_data, *gradInputImages_data;
  inputImages_data = THFloatTensor_data(inputImages);
  gradOutput_data = THFloatTensor_data(gradOutput);
  grids_data = THFloatTensor_data(grids);
  gradGrids_data = THFloatTensor_data(gradGrids);
  gradInputImages_data = THFloatTensor_data(gradInputImages);

  int b, yOut, xOut,zOut;
//  float length;length=sizeof(grids_data)/sizeof(data[0];
  for(b=0; b < batchsize; b++)
  {
    for(yOut=0; yOut < gradOutput_height; yOut++)
    {
      for(xOut=0; xOut < gradOutput_width; xOut++)
      {
        for(zOut=0;zOut<gradOutput_depth;zOut++)
        {        //read the grid
//            real index = b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth+ zOut*grids_strideDepth + 2*grids_strideChannel;
            real xf = grids_data[b*grids_strideBatch+yOut*grids_strideHeight+xOut*grids_strideWidth+zOut*grids_strideDepth+2*grids_strideChannel];
//            real xf = grids_data[0];
//            printf("xf%f\n", xf );
//            printf("a=%f\n", index);
            real yf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth + grids_strideChannel];
//            printf("yf%f\n", yf );
            real zf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth];

            int yInTopLeft, xInTopLeft,zInTopLeft;
            real yWeightTopLeft, xWeightTopLeft, zWeightTopLeft;

            real xcoord = (xf + 1) * (inputImages_width - 1) / 2;
            xInTopLeft = floor(xcoord);
//            printf("xInTopLeft %d\n",xInTopLeft);
            xWeightTopLeft = 1 - (xcoord - xInTopLeft);
//            printf("xWeight %f\n",xWeightTopLeft);
            real ycoord = (yf + 1) * (inputImages_height - 1) / 2;
            yInTopLeft = floor(ycoord);
//            printf("yInTopLeft %d\n",yInTopLeft);
            yWeightTopLeft = 1 - (ycoord - yInTopLeft);


            real zcoord = (yf + 1) * (inputImages_depth - 1) / 2;
            zInTopLeft = floor(zcoord);
//            printf("zInTopLeft %d\n",zInTopLeft);
            zWeightTopLeft = 1 - (zcoord - zInTopLeft);

            const int inTopLeftBackAddress = inputImages_strideBatch * b + inputImages_strideHeight * yInTopLeft + inputImages_strideWidth * xInTopLeft+inputImages_strideDepth*zInTopLeft;
            const int inTopRightBackAddress = inTopLeftBackAddress + inputImages_strideWidth;
            const int inBottomLeftBackAddress = inTopLeftBackAddress + inputImages_strideHeight;
            const int inBottomRightBackAddress = inBottomLeftBackAddress + inputImages_strideWidth;

            const int inTopLeftForeAddress = inTopLeftBackAddress + inputImages_strideDepth;
            const int inTopRightForeAddress = inTopLeftForeAddress + inputImages_strideWidth;
            const int inBottomLeftForeAddress = inTopLeftForeAddress + inputImages_strideHeight;
            const int inBottomRightForeAddress = inBottomLeftForeAddress + inputImages_strideWidth;

            const int gradInputImagesTopLeftBackAddress = gradInputImages_strideBatch * b + gradInputImages_strideHeight * yInTopLeft + gradInputImages_strideWidth * xInTopLeft + gradInputImages_strideDepth*zInTopLeft;
            const int gradInputImagesTopRightBackAddress = gradInputImagesTopLeftBackAddress + gradInputImages_strideWidth;
            const int gradInputImagesBottomLeftBackAddress = gradInputImagesTopLeftBackAddress + gradInputImages_strideHeight;
            const int gradInputImagesBottomRightBackAddress = gradInputImagesBottomLeftBackAddress + gradInputImages_strideWidth;

            const int gradInputImagesTopLeftForeAddress = gradInputImagesBottomRightBackAddress + gradInputImages_strideDepth;
            const int gradInputImagesTopRightForeAddress = gradInputImagesTopLeftForeAddress + gradInputImages_strideWidth;
            const int gradInputImagesBottomLeftForeAddress = gradInputImagesTopLeftForeAddress + gradInputImages_strideHeight;
            const int gradInputImagesBottomRightForeAddress = gradInputImagesBottomLeftForeAddress + gradInputImages_strideWidth;


            const int gradOutputAddress = gradOutput_strideBatch * b + gradOutput_strideHeight * yOut + gradOutput_strideWidth * xOut + gradOutput_depth*zOut;

            real topLeftBackDotProduct = 0;
            real topRightBackDotProduct = 0;
            real bottomLeftBackDotProduct = 0;
            real bottomRightBackDotProduct = 0;
            real topLeftForeDotProduct = 0;
            real topRightForeDotProduct = 0;
            real bottomLeftForeDotProduct = 0;
            real bottomRightForeDotProduct = 0;

            real v=0;
            real inTopLeftBack=0;
            real inTopRightBack=0;
            real inBottomLeftBack=0;
            real inBottomRightBack=0;
            real inTopLeftFore=0;
            real inTopRightFore=0;
            real inBottomLeftFore=0;
            real inBottomRightFore=0;

            // we are careful with the boundaries
            bool topLeftBackIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_width-1 && yInTopLeft >= 0 && yInTopLeft <= inputImages_height-1 && zInTopLeft>=0 && zInTopLeft <=inputImages_depth-1;
            bool topRightBackIsIn = xInTopLeft+1 >= 0 && xInTopLeft+1 <= inputImages_width-1 && yInTopLeft >= 0 && yInTopLeft <= inputImages_height-1 && zInTopLeft >=0 &&zInTopLeft<=inputImages_depth-1 ;
            bool bottomLeftBackIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_width-1 && yInTopLeft+1 >= 0 && yInTopLeft+1 <= inputImages_height-1&& zInTopLeft >=0 &&zInTopLeft<=inputImages_depth-1;
            bool bottomRightBackIsIn = xInTopLeft+1 >= 0 && xInTopLeft+1 <= inputImages_width-1 && yInTopLeft+1 >= 0 && yInTopLeft+1 <= inputImages_height-1&& zInTopLeft >=0 &&zInTopLeft<=inputImages_depth-1;

            bool topLeftForeIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_width-1 && yInTopLeft >= 0 && yInTopLeft <= inputImages_height-1 && zInTopLeft+1>=0 && zInTopLeft+1 <=inputImages_depth-1;
            bool topRightForeIsIn = xInTopLeft+1 >= 0 && xInTopLeft+1 <= inputImages_width-1 && yInTopLeft >= 0 && yInTopLeft <= inputImages_height-1 && zInTopLeft+1 >=0 &&zInTopLeft+1<=inputImages_depth-1 ;
            bool bottomLeftForeIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_width-1 && yInTopLeft+1 >= 0 && yInTopLeft+1 <= inputImages_height-1&& zInTopLeft+1 >=0 &&zInTopLeft+1<=inputImages_depth-1;
            bool bottomRightForeIsIn = xInTopLeft+1 >= 0 && xInTopLeft+1 <= inputImages_width-1 && yInTopLeft+1 >= 0 && yInTopLeft+1 <= inputImages_height-1&& zInTopLeft+1 >=0 &&zInTopLeft+1<=inputImages_depth-1;
            int t;

            for(t=0; t<inputImages_channels; t++)
            {
               real gradOutValue = gradOutput_data[gradOutputAddress + t * gradOutput_strideChannel];
//               printf("gradOutValue %.8f\n",gradOutValue);
               if(topLeftBackIsIn)
               {
                  real inTopLeftBack = inputImages_data[inTopLeftBackAddress + t * inputImages_strideChannel];
                  topLeftBackDotProduct += inTopLeftBack * gradOutValue;
                  if(!onlyGrid) gradInputImages_data[gradInputImagesTopLeftBackAddress + t * gradInputImages_strideChannel] += xWeightTopLeft * yWeightTopLeft*zWeightTopLeft * gradOutValue;
               }

               if(topRightBackIsIn)
               {
                  real inTopRightBack = inputImages_data[inTopRightBackAddress + t * inputImages_strideChannel];
                  topRightBackDotProduct += inTopRightBack * gradOutValue;
                  if(!onlyGrid) gradInputImages_data[gradInputImagesTopRightBackAddress + t * gradInputImages_strideChannel] += (1 - xWeightTopLeft) * yWeightTopLeft *zWeightTopLeft* gradOutValue;
               }

               if(bottomLeftBackIsIn)
               {
                  real inBottomLeftBack = inputImages_data[inBottomLeftBackAddress + t * inputImages_strideChannel];
                  bottomLeftBackDotProduct += inBottomLeftBack * gradOutValue;
                  if(!onlyGrid) gradInputImages_data[gradInputImagesBottomLeftBackAddress + t * gradInputImages_strideChannel] += xWeightTopLeft * (1 - yWeightTopLeft) *zWeightTopLeft* gradOutValue;
               }

               if(bottomRightBackIsIn)
               {
                  real inBottomRightBack = inputImages_data[inBottomRightBackAddress + t * inputImages_strideChannel];
                  bottomRightBackDotProduct += inBottomRightBack * gradOutValue;
                  if(!onlyGrid) gradInputImages_data[gradInputImagesBottomRightBackAddress + t * gradInputImages_strideChannel] += (1 - xWeightTopLeft) * (1 - yWeightTopLeft)*(zWeightTopLeft) * gradOutValue;
               }

                  if(topLeftForeIsIn)
               {
                  real inTopLeftFore = inputImages_data[inTopLeftForeAddress + t * inputImages_strideChannel];
                  topLeftForeDotProduct += inTopLeftFore * gradOutValue;
                  if(!onlyGrid) gradInputImages_data[gradInputImagesTopLeftForeAddress + t * gradInputImages_strideChannel] += xWeightTopLeft * yWeightTopLeft*(1-zWeightTopLeft) * gradOutValue;
               }

               if(topRightForeIsIn)
               {
                  real inTopRightFore = inputImages_data[inTopRightForeAddress + t * inputImages_strideChannel];
                  topRightForeDotProduct += inTopRightFore * gradOutValue;
                  if(!onlyGrid) gradInputImages_data[gradInputImagesTopRightForeAddress + t * gradInputImages_strideChannel] += (1 - xWeightTopLeft) * yWeightTopLeft *(1-zWeightTopLeft)* gradOutValue;
               }

               if(bottomLeftForeIsIn)
               {
                  real inBottomLeftFore = inputImages_data[inBottomLeftForeAddress + t * inputImages_strideChannel];
                  bottomLeftForeDotProduct += inBottomLeftFore * gradOutValue;
                  if(!onlyGrid) gradInputImages_data[gradInputImagesBottomLeftForeAddress + t * gradInputImages_strideChannel] += xWeightTopLeft * (1 - yWeightTopLeft) *(1-zWeightTopLeft)* gradOutValue;
               }

               if(bottomRightForeIsIn)
               {
                  real inBottomRightFore = inputImages_data[inBottomRightForeAddress + t * inputImages_strideChannel];
                  bottomRightForeDotProduct += inBottomRightFore * gradOutValue;
                  if(!onlyGrid) gradInputImages_data[gradInputImagesBottomRightForeAddress + t * gradInputImages_strideChannel] += (1-xWeightTopLeft)*(1-yWeightTopLeft)*(1-zWeightTopLeft)*gradOutValue;
               }
            }
            xf = -1*(yWeightTopLeft*zWeightTopLeft*topLeftBackDotProduct -yWeightTopLeft*zWeightTopLeft*topRightBackDotProduct+(1-yWeightTopLeft)*zWeightTopLeft*bottomLeftBackDotProduct
                 -(1-yWeightTopLeft)*zWeightTopLeft*bottomRightBackDotProduct+yWeightTopLeft*(1-zWeightTopLeft)*topLeftForeDotProduct - yWeightTopLeft*(1-zWeightTopLeft)*topRightForeDotProduct
                 +(1-yWeightTopLeft)*(1-zWeightTopLeft)*bottomLeftForeDotProduct-(1-yWeightTopLeft)*(1-zWeightTopLeft)*bottomRightForeDotProduct);

            yf = -1*(xWeightTopLeft*zWeightTopLeft*topLeftBackDotProduct + (1-xWeightTopLeft)*zWeightTopLeft*topRightBackDotProduct-xWeightTopLeft*zWeightTopLeft*bottomLeftBackDotProduct
                 +(1-xWeightTopLeft)*zWeightTopLeft*bottomRightBackDotProduct +xWeightTopLeft*(1-zWeightTopLeft)*topLeftForeDotProduct +(1-xWeightTopLeft)*(1-zWeightTopLeft)*topRightForeDotProduct
                 -xWeightTopLeft*(1-zWeightTopLeft)*bottomLeftForeDotProduct-(1-xWeightTopLeft)*(1-zWeightTopLeft)*bottomRightForeDotProduct);
            zf = -1*(xWeightTopLeft*yWeightTopLeft*topLeftBackDotProduct+(1-xWeightTopLeft)*yWeightTopLeft*topRightBackDotProduct+xWeightTopLeft*(1-yWeightTopLeft)*bottomLeftBackDotProduct
                 +(1-xWeightTopLeft)*(1-yWeightTopLeft)*bottomRightBackDotProduct-xWeightTopLeft*yWeightTopLeft*topLeftForeDotProduct-(1-xWeightTopLeft)*yWeightTopLeft*topRightForeDotProduct
                 -xWeightTopLeft*(1-yWeightTopLeft)*bottomLeftForeDotProduct-(1-xWeightTopLeft)*(1-yWeightTopLeft)*bottomRightForeDotProduct);


//            real value = xf;
            gradGrids_data[b*gradGrids_strideBatch + yOut*gradGrids_strideHeight + xOut*gradGrids_strideWidth +2* gradGrids_strideChannel] = zf * (inputImages_width-1) / 2;
//            printf("xf %f\n", value);
            gradGrids_data[b*gradGrids_strideBatch + yOut*gradGrids_strideHeight + xOut*gradGrids_strideWidth+gradGrids_strideChannel] = yf * (inputImages_height-1) / 2;
            gradGrids_data[b*gradGrids_strideBatch + yOut*gradGrids_strideHeight + xOut*gradGrids_strideWidth] = xf * (inputImages_height-1) / 2;

        }
      }
    }
  }

  return 1;
}




