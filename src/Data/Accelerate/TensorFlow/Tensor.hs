{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# LANGUAGE ExplicitNamespaces #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Data.Accelerate.TensorFlow.Tensor where

import Data.Accelerate.TensorFlow.Type
import Data.Array.Accelerate.Representation.Shape
import Data.Array.Accelerate.Array.Buffer
import Data.Array.Accelerate.Array.Unique (UniqueArray(UniqueArray), withUniqueArrayPtr)
import Data.Array.Accelerate.Lifetime
import GHC.ForeignPtr
import Control.Monad.IO.Class (liftIO)
import Data.Array.Accelerate.Type (ScalarType (..), SingleType (NumSingleType), NumType (IntegralNumType, FloatingNumType), IntegralType (..), FloatingType (..), VectorType (..))

import qualified TensorFlow.Ops                                     as TF
import qualified TensorFlow.Core                                    as TF
import qualified TensorFlow.Tensor                                  as TF
import qualified TensorFlow.Types                                   as TF
import qualified TensorFlow.Session                                 as TF
import qualified TensorFlow.Internal.FFI as FFI
import qualified Data.Vector as V
import Data.Int
import qualified TensorFlow.GenOps.Core                             as TF hiding (shape, placeholder)
import Foreign (Ptr, castPtr, Word8)
import Unsafe.Coerce (unsafeCoerce)
import Data.Array.Accelerate.Analysis.Match ( type (:~:)(Refl), matchScalarType )
import qualified Data.Functor.Identity as TF
import qualified Data.Vector.Storable as S
import Text.Printf

toTFShape :: ShapeR sh -> sh -> TF.Shape
toTFShape shR sh = TF.Shape $ fromIntegral <$> shapeToList shR sh

fromBuffer :: (TF.TensorType t, S.Storable t, TF.TensorDataType S.Vector t) =>
  ShapeR sh -> 
  ScalarType t -> 
  sh -> 
  Buffer t ->
  TF.Session (S.Vector t)
fromBuffer shR t sh buffer = do
  let shape = toTFShape shR sh
  tensorData <- toTensorData t shape (size shR sh) buffer
  TF.runSession $ do
    x <- TF.placeholder shape
    let feeds = [ TF.feed x tensorData ]
    TF.runWithFeeds feeds $ TF.identity x

toTensorData :: (TF.TensorType t, S.Storable t, TF.TensorDataType S.Vector t) =>
  ScalarType t -> 
  TF.Shape ->
  Int ->
  Buffer t ->
  TF.Session (TF.TensorData t)
toTensorData t sh n buffer = do 
  let vec = S.fromList (bufferToList t n buffer)
  return $ TF.encodeTensorData sh vec

toBuffer :: ScalarType t -> IO (V.Vector t) -> IO (Buffer t)
toBuffer t v = undefined

-- 1 Data.Array.Accelerate.AST.Execute
-- executeAfunSchedule :: GFunctionR t -> sched kernel () (Scheduled sched t) -> IOFun (Scheduled sched t)
